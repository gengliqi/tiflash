# Reusable Async gRPC Design for TiFlash

## Background

Today TiFlash already has the key low-level pieces needed for efficient async gRPC:

- `GRPCKickTag` gives us an immediate completion-queue wake-up without using a timer.
- `GRPCSendQueue<T>` bridges business threads and the gRPC write side.
- `GRPCRecvQueue<T>` bridges the gRPC read side and business threads.

This is already enough to implement one async server-streaming RPC well:

- server side: `EstablishCallData`
- client side: `AsyncRequestHandler<RPCContext>`

However, the current abstraction boundary is still too low. The queues are reusable, but every new async RPC still needs a handwritten call object, handwritten state machine, handwritten lifecycle rules, and handwritten CQ event dispatch.

That is manageable for one RPC, but it becomes expensive once we want to support more async RPCs, especially:

- unary RPCs
- server-streaming RPCs
- client-streaming RPCs
- bidirectional-streaming RPCs

The goal of this note is to define a reusable async gRPC framework for TiFlash that keeps the current performance properties and makes future async RPCs much cheaper to build.

## Current Reference Points

The proposal in this note is grounded on the current implementation points below:

- `dbms/src/Common/GRPCKickTag.h`
- `dbms/src/Common/GRPCQueue.h`
- `dbms/src/Flash/EstablishCall.h`
- `dbms/src/Flash/EstablishCall.cpp`
- `dbms/src/Flash/Mpp/AsyncRequestHandler.h`
- `dbms/src/Flash/Mpp/GRPCReceiverContext.h`
- `dbms/src/Flash/Mpp/GRPCReceiverContext.cpp`
- `dbms/src/Server/FlashGrpcServerHolder.cpp`

## Design Goals

1. Reuse the common mechanics, not just the queue classes.
2. Keep all gRPC operations on CQ poller threads.
3. Preserve the current zero-timer immediate wake-up path based on `kick`.
4. Support both server-side and client-side async calls.
5. Support unary, server-streaming, client-streaming, and bidi-streaming.
6. Allow incremental migration from the current code.
7. Avoid a “one giant generic queue” that hurts the hot path.

## Non-Goals

1. Switching TiFlash to gRPC callback API in this iteration.
2. Replacing all existing sync RPCs.
3. Hiding all business states inside a generic framework.
4. Forcing send-side and recv-side backpressure into the exact same internal data structure.

## What Is Already Generic in the Current Code

The current implementation already exposes three important invariants:

### 1. One outstanding operation per direction

For one stream endpoint, gRPC only allows:

- at most one outstanding `Read`
- at most one outstanding `Write`

This is the real reason why `EstablishCallData` and `AsyncRequestHandler` need state machines.

### 2. Business threads must not call stream read/write directly

`GRPCSendQueue<T>` and `GRPCRecvQueue<T>` solve the same problem in opposite directions:

- write side: business threads produce data, CQ thread issues `Write`
- read side: CQ thread receives data, business threads consume it

The important property is not “queue” by itself. The important property is:

- arm a waiter in one thread
- wake it from another thread
- never lose that notification

### 3. `kick` is the cheapest wake-up primitive we have

`GRPCKickTag` is effectively an immediate `Alarm`, implemented by submitting an empty batch on the underlying `grpc_call`. This is a good primitive and should remain the core wake-up path.

## Why the Current Shape Does Not Scale

### 1. The state machine is mixed with business logic

`EstablishCallData` currently owns all of the following at once:

- server accept/spawn logic
- tunnel wait logic
- write-loop logic
- error packet logic
- finish logic
- metrics
- self-destruction rules

This makes it hard to reuse for another server-streaming RPC, even if the control flow is almost the same.

### 2. Client-side async read path duplicates the same CQ mechanics

`AsyncRequestHandler<RPCContext>` reimplements:

- start event handling
- retry/alarm handling
- read completion handling
- “queue full then arm tag” handling
- finish event handling

The business logic differs, but the CQ/event skeleton is almost the same.

### 3. The current “single tag + single stage enum” model does not scale to bidi

The current pattern works because each call only has one active direction:

- `EstablishCallData` only writes
- `AsyncRequestHandler` only reads

For bidirectional streaming this is no longer enough. Read and write may both be outstanding at the same time. A single `this` tag and a single stage enum are not a good fit for that shape.

This is the main structural reason to move from “call object is the only tag” to “call object owns several fixed tags”.

## Proposed Architecture

The proposal is to split the problem into four reusable layers.

### Layer 1: Typed CQ Tags

Introduce a reusable tag type that separates:

- the owner call
- the event kind
- the wake-up reason

Suggested shape:

```cpp
enum class AsyncRpcEventKind
{
    Request,
    Start,
    Read,
    Write,
    WritesDone,
    Finish,
    Alarm,
    WakeRead,
    WakeWrite,
    WakeExternal,
};

enum class AsyncRpcWakeReason
{
    Ready,
    Closed,
    Cancelled,
    ExternalReady,
};

class AsyncRpcTag final : public grpc::internal::CompletionQueueTag
{
public:
    AsyncRpcCallBase * owner;
    AsyncRpcEventKind kind;
    AsyncRpcWakeReason wake_reason;
    grpc_call * call;

    void kick(AsyncRpcWakeReason reason);
    bool FinalizeResult(void ** tag, bool * ok) override;
};
```

Each call object embeds a fixed set of tags:

- `request_tag`
- `start_tag`
- `read_tag`
- `write_tag`
- `writes_done_tag`
- `finish_tag`
- `alarm_tag`
- `wake_read_tag`
- `wake_write_tag`
- `wake_external_tag`

Key properties:

- no heap allocation per message
- no heap allocation per CQ event
- separate tags allow concurrent read and write in bidi
- CQ dispatch becomes generic and easy to reason about

`GRPCKickTag` can be kept as a compatibility wrapper first, then folded into this new typed tag.

### Layer 2: Flow Bridges

Do not over-generalize the queue internals. Instead, generalize the interface and keep two specialized implementations.

Suggested interfaces:

```cpp
enum class FlowResult
{
    Ready,
    NeedWait,
    Finished,
    Cancelled,
};

template <typename T>
class OutboundFlowBridge;

template <typename T>
class InboundFlowBridge;
```

#### `OutboundFlowBridge<T>`

This is the reusable form of the current `GRPCSendQueue<T>`.

Purpose:

- business threads push outbound messages
- CQ thread drains them and issues `Write`

API sketch:

```cpp
FlowResult tryNextOrArm(T & value, AsyncRpcTag * wake_tag);
MPMCQueueResult push(T && value);
MPMCQueueResult forcePush(T && value);
bool finish();
bool cancelWith(const String & reason);
```

#### `InboundFlowBridge<T>`

This is the reusable form of the current `GRPCRecvQueue<T>`.

Purpose:

- CQ thread receives inbound messages
- business threads pop them later

API sketch:

```cpp
FlowResult pushOrArm(T && value, AsyncRpcTag * wake_tag);
MPMCQueueResult pop(T & value);
MPMCQueueResult tryPop(T & value);
bool finish();
bool cancelWith(const String & reason);
```

Important point: the two directions should keep their current specialized lock/data-layout strategy. The abstraction should be at the API level, not by forcing them into one slow universal class.

That means the first step can be very small:

- keep current `GRPCSendQueue<T>` and `GRPCRecvQueue<T>` implementations
- add adapters or rename them to `OutboundFlowBridge<T>` / `InboundFlowBridge<T>`
- move future generic call code to depend on the bridge interface only

### Layer 3: Direction Pumps

Build two reusable “pumps” on top of tags and bridges.

#### Read Pump

The read pump handles the standard async read loop:

1. issue one `Read`
2. when read completes, hand data to `InboundFlowBridge`
3. if bridge is full, arm `wake_read_tag` and stop
4. when awakened, issue next `Read`
5. on EOF, move to finish path

#### Write Pump

The write pump handles the standard async write loop:

1. ask `OutboundFlowBridge` for the next message
2. if empty, arm `wake_write_tag` and stop
3. if a message exists, issue one `Write`
4. when write completes, continue draining
5. on finish/cancel, move to finish path

Why this split is useful:

- server-streaming only needs the write pump
- client-streaming mainly needs the read pump
- bidi needs both pumps simultaneously
- the business handler no longer needs to manage “WAIT_READ / WAIT_PUSH / WAIT_WRITE / WAIT_IN_QUEUE” directly

### Layer 4: Generic Call Shell

Introduce a generic call shell that owns:

- the grpc context and stream object
- the fixed tags
- the alarm
- read/write pump state
- finish state
- lifetime management

Suggested shape:

```cpp
template <class Traits, class Handler>
class AsyncRpcCall final;
```

Where:

- `Traits` describes the gRPC shape and actual gRPC types
- `Handler` provides business-specific behavior

#### `Traits` responsibilities

`Traits` should define:

- `Request`
- `Response`
- gRPC responder/reader/reader-writer type
- whether the call is server-side or client-side
- whether the server accept path exposes an eagerly available initial request object
- whether the call has read side, write side, or both
- how to start/request/read/write/finish/writes-done

For example:

```cpp
struct EstablishMppServerStreamTraits
{
    static constexpr bool is_server = true;
    static constexpr bool has_read = false;
    static constexpr bool has_write = true;
    static constexpr bool has_writes_done = false;

    using Service = AsyncFlashService;
    using Request = mpp::EstablishMPPConnectionRequest;
    using Response = mpp::MPPDataPacket;
    using Stream = grpc::ServerAsyncWriter<Response>;

    static void request(...);
    static void write(Stream &, const Response &, AsyncRpcTag *);
    static void finish(Stream &, const grpc::Status &, AsyncRpcTag *);
};
```

#### `Handler` responsibilities

`Handler` should only own business logic:

- request validation
- how to obtain the bridge or data source
- retry policy
- whether to wait for an external resource
- how to build final status / final error packet
- metrics specific to this RPC

The handler should not own CQ dispatch boilerplate.

## Generic Server Acceptor

Server-side async RPCs all share the same accept pattern:

1. submit `RequestXxx(...)`
2. when request arrives, immediately spawn the next acceptor
3. turn the current instance into an active call
4. destroy the call after `Finish` completes

This should be extracted into a common server acceptor/factory instead of living inside each call type.

Suggested shape:

```cpp
template <class Traits, class HandlerFactory>
class AsyncServerRpcFactory;
```

Responsibilities:

- preallocate acceptors for each CQ
- own the `request_tag`
- spawn next acceptor automatically
- pass the accepted call, plus an optional initial request object, to `AsyncRpcCall<Traits, Handler>`

This would remove the `EstablishCallData::spawn(...)` boilerplate and make `FlashGrpcServerHolder` independent of a specific RPC.

## Generic Client Launcher

Client-side async calls also share a common start pattern:

1. pick a CQ from `GRPCCompletionQueuePool`
2. issue async start call
3. once the start event completes, bind the call handle to wake tags
4. start the read/write pump(s)
5. optionally retry with `Alarm`

Suggested shape:

```cpp
template <class Traits, class Handler>
class AsyncClientRpcLauncher;
```

This is the reusable shell that should replace most of the current `AsyncRequestHandler<RPCContext>` mechanics.

## How Each RPC Shape Fits

### Unary RPC

Server side:

- accept request
- run business logic
- issue one response and `Finish`

Client side:

- start call
- wait for one completion
- `Finish`

The framework can use the same tag shell but a degenerate pump configuration.

### Server-Streaming RPC

This matches `EstablishMPPConnection`.

Needed pieces:

- server acceptor
- write pump
- optional external wait before write pump starts
- final `Finish`

This shape should become almost entirely reusable.

### Client-Streaming RPC

Needed pieces:

- read pump
- business-side accumulation or inbound bridge
- final one-shot response
- `Finish`

Unlike server-streaming, server-side client-streaming may not have an eagerly materialized initial request payload at accept time. The framework should therefore let traits describe whether the accept path provides an initial request object or whether the read pump owns the full request stream from the beginning.

### Bidirectional-Streaming RPC

Needed pieces:

- read pump
- write pump
- separate read and write tags
- optional `WritesDone`
- coordinated close semantics

Like client-streaming, server-side bidi may not have an eagerly materialized initial request object. This is the strongest reason to introduce fixed typed tags. A bidi call must treat read-side and write-side progress as partially independent state machines.

## External Waits and Retry

Not every async RPC is “start and immediately read/write”.

`EstablishMPPConnection` has a real external wait:

- wait for the target MPP task/tunnel to become visible

Future RPCs may also wait for:

- scheduler quota
- memory budget
- remote metadata
- retry backoff

The framework should support this through a generic external wake path:

- call enters `WAIT_EXTERNAL`
- handler stores `wake_external_tag` in some manager-owned registry
- external code kicks that tag when the resource becomes ready
- alarm-based timeout/backoff is also routed through typed tags

This keeps external waiting reusable without mixing it into the read/write pump logic.

## Threading Model

The framework should enforce the following model explicitly:

1. Only CQ poller threads issue gRPC operations.
2. Business threads only interact through flow bridges or external wake handles.
3. Every armed wake tag has exactly one owner at a time.
4. A call may destroy itself only after all armed tags have been disarmed by:
   - a CQ event
   - bridge cancellation/finish
   - shutdown cleanup

This is already how the current code works informally. The new framework should make it explicit and debuggable.

## Performance Considerations

The design is intended to preserve the current performance profile.

### 1. No per-message heap allocation in the framework itself

All CQ tags are embedded in the call object.

### 2. Keep `kick` instead of replacing it with `Alarm`

Immediate wake-up should still use the current empty-batch trick. `Alarm` stays only for timeout/retry.

### 3. Do not unify send/recv bridge internals too aggressively

The current send-side and recv-side queues have different contention patterns:

- send side stores one waiter tag
- recv side may temporarily store multiple pending data+tag pairs

That specialization should remain.

### 4. No gRPC API on business threads

This avoids lock inversion and preserves the nice property that all stream operations stay serialized by the CQ model.

### 5. Prefer CRTP / traits over virtual-heavy runtime polymorphism on hot path

One virtual dispatch from gRPC into the tag is unavoidable, but the rest of the framework can stay compile-time bound.

## Migration Plan

The migration should be incremental.

### Step 1: Extract reusable primitives without changing behavior

- introduce typed async tag
- add bridge adapters over `GRPCSendQueue<T>` / `GRPCRecvQueue<T>`
- add a generic server acceptor shell

At this step, no business logic changes are required.

### Step 2: Migrate `EstablishMPPConnection` server side

Turn `EstablishCallData` into:

- a small server-streaming handler
- plus a reusable `AsyncRpcCall` shell

The business-specific part that should remain custom is mainly:

- tunnel lookup / wait
- error packet generation
- MPP-specific metrics

The generic write pump should absorb:

- `WAIT_WRITE`
- `WAIT_IN_QUEUE`
- empty queue arming
- wake-up and continue-write logic

### Step 3: Migrate `AsyncRequestHandler<RPCContext>`

Turn it into:

- a generic client launcher + read pump
- plus an MPP-specific handler for retry and close policy

The generic read pump should absorb:

- `WAIT_MAKE_READER`
- `WAIT_READ`
- `WAIT_PUSH_TO_QUEUE`
- queue-full arming and resume logic

### Step 4: Add first-class helpers for unary and bidi

Once server-streaming and client-read-streaming are on the same shell, add:

- unary helper
- client-stream helper
- bidi helper

The bidi helper should be designed last, after fixed tags and dual-pump semantics are proven.

## Suggested File Layout

One possible layout is:

- `dbms/src/Common/grpc/AsyncRpcTag.h`
- `dbms/src/Common/grpc/AsyncFlowBridge.h`
- `dbms/src/Common/grpc/AsyncRpcCall.h`
- `dbms/src/Common/grpc/AsyncServerRpcFactory.h`
- `dbms/src/Common/grpc/AsyncClientRpcLauncher.h`

During migration we can keep compatibility wrappers:

- `GRPCKickTag.h`
- `GRPCQueue.h`

and gradually move new code to the new headers.

## Expected Outcome

After this refactor, adding a new async RPC should mostly mean:

1. define the traits for the gRPC method shape
2. define the business handler
3. optionally choose one or two bridges
4. register the factory/launcher

Instead of re-implementing:

- a call object
- a CQ state machine
- wake-up logic
- queue-arm logic
- finish/lifetime boilerplate

## Recommendation

The right abstraction boundary is:

- keep `kick` and the current queue semantics
- extract CQ tags, pumps, and call lifecycle into a reusable framework
- treat bidi as the design target that forces multiple fixed tags

In other words, the main reuse opportunity is not “make one bigger queue class”. The main reuse opportunity is “separate generic CQ/pump/lifecycle mechanics from RPC-specific business logic”.

That approach keeps the current performance characteristics while making future async RPCs much easier to add.
