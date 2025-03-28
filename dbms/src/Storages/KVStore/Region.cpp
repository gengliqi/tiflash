// Copyright 2023 PingCAP, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <Common/FmtUtils.h>
#include <Common/ProfileEvents.h>
#include <Common/Stopwatch.h>
#include <Common/TiFlashMetrics.h>
#include <Storages/DeltaMerge/DeltaMergeInterfaces.h>
#include <Storages/KVStore/Decode/TiKVRange.h>
#include <Storages/KVStore/FFI/ProxyFFI.h>
#include <Storages/KVStore/KVStore.h>
#include <Storages/KVStore/Region.h>
#include <Storages/KVStore/TMTContext.h>
#include <Storages/KVStore/Types.h>
#include <common/logger_useful.h>

#include <ext/scope_guard.h>
#include <memory>

extern std::atomic<Int64> real_rss;
std::atomic<UInt64> tranquil_time_rss;

namespace DB
{
RegionData::WriteCFIter Region::removeDataByWriteIt(const RegionData::WriteCFIter & write_it)
{
    return data.removeDataByWriteIt(write_it);
}

std::optional<RegionDataReadInfo> Region::readDataByWriteIt(
    const RegionData::ConstWriteCFIter & write_it,
    bool need_value,
    bool hard_error)
{
    try
    {
        return data.readDataByWriteIt(write_it, need_value, id(), appliedIndex(), hard_error);
    }
    catch (DB::Exception & e)
    {
        e.addMessage(fmt::format("(applied_term: {})", appliedIndexTerm()));
        throw;
    }
}

LockInfoPtr Region::getLockInfo(const RegionLockReadQuery & query) const
{
    return data.getLockInfo(query);
}

void Region::insertDebug(const std::string & cf, TiKVKey && key, TiKVValue && value, DupCheck mode)
{
    std::unique_lock<std::shared_mutex> lock(mutex);
    doInsert(NameToCF(cf), std::move(key), std::move(value), mode);
}

void Region::insertFromSnap(TMTContext & tmt, const std::string & cf, TiKVKey && key, TiKVValue && value, DupCheck mode)
{
    insertFromSnap(tmt, NameToCF(cf), std::move(key), std::move(value), mode);
}

void Region::insertFromSnap(TMTContext & tmt, ColumnFamilyType type, TiKVKey && key, TiKVValue && value, DupCheck mode)
{
    std::unique_lock<std::shared_mutex> lock(mutex);
    doInsert(type, std::move(key), std::move(value), mode);
    maybeWarnMemoryLimitByTable(tmt, "snapshot");
}

RegionDataMemDiff Region::doInsert(ColumnFamilyType type, TiKVKey && key, TiKVValue && value, DupCheck mode)
{
    if unlikely (getClusterRaftstoreVer() == RaftstoreVer::V2)
    {
        if (type == ColumnFamilyType::Write)
        {
            if (orphanKeysInfo().observeKeyFromNormalWrite(key))
            {
                // We can't assert the key exists in write_cf here,
                // since it may be already written into DeltaTree.
                return RegionDataMemDiff{};
            }
        }
    }
    return data.insert(type, std::move(key), std::move(value), mode);
}

void Region::removeDebug(const std::string & cf, const TiKVKey & key)
{
    std::unique_lock<std::shared_mutex> lock(mutex);
    doRemove(NameToCF(cf), key);
}

void Region::doRemove(ColumnFamilyType type, const TiKVKey & key)
{
    data.remove(type, key);
}

void Region::clearAllData()
{
    std::unique_lock lock(mutex);
    data.assignRegionData(RegionData());
}

UInt64 Region::appliedIndex() const
{
    return meta.appliedIndex();
}

UInt64 Region::appliedIndexTerm() const
{
    return meta.appliedIndexTerm();
}

void Region::setApplied(UInt64 index, UInt64 term)
{
    std::unique_lock lock(mutex);
    meta.setApplied(index, term);
}

RegionPtr Region::splitInto(RegionMeta && meta)
{
    RegionPtr new_region = std::make_shared<Region>(std::move(meta), proxy_helper);

    const auto range = new_region->getRange();
    data.splitInto(range->comparableKeys(), new_region->data);

    return new_region;
}

std::string Region::getDebugString() const
{
    const auto & meta_snap = meta.dumpRegionMetaSnapshot();
    return fmt::format(
        "[region_id={} index={} {}table_id={} ver={} conf_ver={} state={} peer={} range={}]",
        id(),
        meta.appliedIndex(),
        ((keyspace_id == NullspaceID) ? "" : fmt::format("keyspace={} ", keyspace_id)),
        mapped_table_id,
        meta_snap.ver,
        meta_snap.conf_ver,
        raft_serverpb::PeerState_Name(peerState()),
        meta_snap.peer.ShortDebugString(),
        getRange()->toDebugString());
}

std::string Region::toString(bool dump_status) const
{
    return meta.toString(dump_status);
}

RegionID Region::id() const
{
    return meta.regionId();
}

bool Region::isPendingRemove() const
{
    return peerState() == raft_serverpb::PeerState::Tombstone;
}

bool Region::isMerging() const
{
    return peerState() == raft_serverpb::PeerState::Merging;
}

void Region::setPendingRemove()
{
    setPeerState(raft_serverpb::PeerState::Tombstone);
}

void Region::setStateApplying()
{
    setPeerState(raft_serverpb::PeerState::Applying);
    snapshot_event_flag++;
}

raft_serverpb::PeerState Region::peerState() const
{
    return meta.peerState();
}

size_t Region::dataSize() const
{
    return data.dataSize();
}

size_t Region::totalSize() const
{
    return data.totalSize() + sizeof(RegionMeta);
}

size_t Region::writeCFCount() const
{
    std::shared_lock<std::shared_mutex> lock(mutex);
    return data.writeCF().getSize();
}

std::string Region::dataInfo() const
{
    std::shared_lock<std::shared_mutex> lock(mutex);

    FmtBuffer buff;
    buff.append("[");
    auto write_size = data.writeCF().getSize();
    auto lock_size = data.lockCF().getSize();
    auto default_size = data.defaultCF().getSize();
    if (write_size)
        buff.fmtAppend("write {} ", write_size);
    if (lock_size)
        buff.fmtAppend("lock {} ", lock_size);
    if (default_size)
        buff.fmtAppend("default {} ", default_size);
    buff.append("]");
    return buff.toString();
}

std::pair<UInt64, UInt64> Region::getRaftLogEagerGCRange() const
{
    std::unique_lock lock(mutex);
    auto applied_index = appliedIndex();
    return {eager_truncated_index, applied_index};
}

void Region::updateRaftLogEagerIndex(UInt64 new_truncate_index)
{
    std::unique_lock lock(mutex);
    eager_truncated_index = new_truncate_index;
}

UInt64 Region::lastRestartLogApplied() const
{
    return last_restart_log_applied;
}

UInt64 Region::lastCompactLogApplied() const
{
    return last_compact_log_applied;
}

void Region::setLastCompactLogApplied(UInt64 new_value) const
{
    last_compact_log_applied = new_value;
}

// Everytime the region is persisted, we update the `last_compact_log_applied`
void Region::updateLastCompactLogApplied(const RegionTaskLock &) const
{
    const UInt64 current_applied_index = appliedIndex();
    if (last_compact_log_applied != 0)
    {
        UInt64 gap = current_applied_index > last_compact_log_applied //
            ? current_applied_index - last_compact_log_applied
            : 0;
        GET_METRIC(tiflash_raft_raft_log_gap_count, type_applied_index).Observe(gap);
    }
    last_compact_log_applied = current_applied_index;
}

ImutRegionRangePtr Region::getRange() const
{
    return meta.getRange();
}

RaftstoreVer Region::getClusterRaftstoreVer()
{
    // In non-debug/test mode, we should assert the proxy_ptr be always not null.
    if (likely(proxy_helper != nullptr))
    {
        if (likely(proxy_helper->fn_get_cluster_raftstore_version))
        {
            // Make debug funcs happy.
            return proxy_helper->fn_get_cluster_raftstore_version(proxy_helper->proxy_ptr, 0, 0);
        }
    }
    return RaftstoreVer::Uncertain;
}

UInt64 Region::version() const
{
    return meta.version();
}

UInt64 Region::confVer() const
{
    return meta.confVer();
}

void Region::assignRegion(Region && new_region)
{
    std::unique_lock<std::shared_mutex> lock(mutex);

    data.assignRegionData(std::move(new_region.data));
    meta.assignRegionMeta(std::move(new_region.meta));
    meta.notifyAll();
    eager_truncated_index = meta.truncateIndex();
}

/// try to clean illegal data because of feature `compaction filter`
void Region::tryCompactionFilter(const Timestamp safe_point)
{
    if (size_t del_write = data.tryCompactionFilter(safe_point); del_write)
    {
        LOG_INFO(log, "delete {} records in write cf for region_id={}", del_write, meta.regionId());
    }
}

RegionMetaSnapshot Region::dumpRegionMetaSnapshot() const
{
    return meta.dumpRegionMetaSnapshot();
}

Region::Region(DB::RegionMeta && meta_, const TiFlashRaftProxyHelper * proxy_helper_)
    : meta(std::move(meta_))
    , eager_truncated_index(meta.truncateIndex())
    , log(Logger::get())
    , keyspace_id(meta.getRange()->getKeyspaceID())
    , mapped_table_id(meta.getRange()->getMappedTableID())
    , proxy_helper(proxy_helper_)
{
    GET_METRIC(tiflash_raft_classes_count, type_region).Increment(1);
}

Region::~Region()
{
    GET_METRIC(tiflash_raft_classes_count, type_region).Decrement();
}

void Region::setPeerState(raft_serverpb::PeerState state)
{
    meta.setPeerState(state);
    meta.notifyAll();
}

metapb::Region Region::cloneMetaRegion() const
{
    return meta.cloneMetaRegion();
}
const metapb::Region & Region::getMetaRegion() const
{
    return meta.getMetaRegion();
}
raft_serverpb::MergeState Region::cloneMergeState() const
{
    return meta.cloneMergeState();
}
const raft_serverpb::MergeState & Region::getMergeState() const
{
    return meta.getMergeState();
}

std::pair<size_t, size_t> Region::getApproxMemCacheInfo() const
{
    return {
        approx_mem_cache_rows.load(std::memory_order_relaxed),
        approx_mem_cache_bytes.load(std::memory_order_relaxed),
    };
}

void Region::cleanApproxMemCacheInfo() const
{
    approx_mem_cache_rows = 0;
    approx_mem_cache_bytes = 0;
}

void Region::mergeDataFrom(const Region & other)
{
    this->data.mergeFrom(other.data);
    this->data.orphan_keys_info.mergeFrom(other.data.orphan_keys_info);
}

void Region::observeLearnerReadEvent(Timestamp read_tso) const
{
    auto ori = last_observed_read_tso.load();
    if (read_tso > ori)
    {
        // Do not retry if failed, though may lost some update here, however the total read_tso can advance.
        last_observed_read_tso.compare_exchange_strong(ori, read_tso);
    }
}

Timestamp Region::getLastObservedReadTso() const
{
    return last_observed_read_tso.load();
}

void Region::setRegionTableCtx(RegionTableCtxPtr ctx) const
{
    data.setRegionTableCtx(ctx);
}

void Region::maybeWarnMemoryLimitByTable(TMTContext & tmt, const char * from)
{
    // If there are data flow in, we will check if the memory is exhausted.
    auto limit = tmt.getKVStore()->getKVStoreMemoryLimit();
    size_t current = real_rss.load() > 0 ? real_rss.load() : 0;
    if unlikely (limit == 0 || current == 0)
        return;
    /// Region management such as split/merge doesn't change the memory consumed by a table in KVStore.
    /// The only cases memory is reduced in a table is removing regions, applying snaps and commiting txns.
    /// The only cases memory is increased in a table is inserting kv pairs and applying snaps.
    /// So, we only print once for a table, until one memory reduce event will happen.
    if unlikely (current >= limit * 0.9)
    {
        auto table_size = getRegionTableSize();
        auto grown_memory = current > tranquil_time_rss ? current - tranquil_time_rss : 0;
        // 15% of the total non-tranquil-time memory, but not exceed 10GB.
        auto table_memory_limit = std::min(grown_memory * 0.15, 10 * 1024ULL * 1024 * 1024);
        if (grown_memory && table_size > table_memory_limit)
        {
            if (!setRegionTableWarned(true))
            {
                // If it is the first time.
#ifdef DBMS_PUBLIC_GTEST
                tmt.getKVStore()->debug_memory_limit_warning_count++;
#endif
                LOG_INFO(
                    log,
                    "Memory limit exceeded, current={} limit={} table_limit={} table_in_mem_size={} table_id={} "
                    "keyspace={} "
                    "region_id={} from={}",
                    current,
                    limit,
                    table_memory_limit,
                    table_size,
                    mapped_table_id,
                    keyspace_id,
                    id(),
                    from);
            }
        }
    }
}

void Region::resetWarnMemoryLimitByTable() const
{
    setRegionTableWarned(false);
}

} // namespace DB
