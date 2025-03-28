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

#include <AggregateFunctions/registerAggregateFunctions.h>
#include <Client/Connection.h>
#include <Common/Config/ConfigProcessor.h>
#include <Common/Exception.h>
#include <Common/NetException.h>
#include <Common/ShellCommand.h>
#include <Common/Stopwatch.h>
#include <Common/StringUtils/StringUtils.h>
#include <Common/Throttler.h>
#include <Common/TiFlashBuildInfo.h>
#include <Common/UnicodeBar.h>
#include <Common/formatReadable.h>
#include <Common/typeid_cast.h>
#include <Core/QueryProcessingStage.h>
#include <Core/Types.h>
#include <DataStreams/AsynchronousBlockInputStream.h>
#include <Functions/registerFunctions.h>
#include <IO/Buffer/ReadBufferFromFileDescriptor.h>
#include <IO/Buffer/ReadBufferFromMemory.h>
#include <IO/Buffer/ReadBufferFromString.h>
#include <IO/Buffer/WriteBufferFromFile.h>
#include <IO/Buffer/WriteBufferFromFileDescriptor.h>
#include <IO/ReadHelpers.h>
#include <IO/WriteHelpers.h>
#include <Interpreters/Context.h>
#include <Parsers/ASTIdentifier.h>
#include <Parsers/ASTInsertQuery.h>
#include <Parsers/ASTLiteral.h>
#include <Parsers/ASTQueryWithOutput.h>
#include <Parsers/ASTSelectWithUnionQuery.h>
#include <Parsers/ASTSetQuery.h>
#include <Parsers/ASTUseQuery.h>
#include <Parsers/ParserQuery.h>
#include <Parsers/formatAST.h>
#include <Parsers/parseQuery.h>
#include <Poco/File.h>
#include <Poco/Util/Application.h>
#include <WindowFunctions/registerWindowFunctions.h>
#include <boost_wrapper/program_options.h>
#include <common/find_symbols.h>
#include <fcntl.h>
#include <port/unistd.h>
#include <signal.h>
#include <stdlib.h>

#include <algorithm>
#include <boost/algorithm/string/replace.hpp>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <unordered_set>

#include "InterruptListener.h"

/// http://en.wikipedia.org/wiki/ANSI_escape_code

/// Similar codes \e[s, \e[u don't work in VT100 and Mosh.
#define SAVE_CURSOR_POSITION "\e7"
#define RESTORE_CURSOR_POSITION "\e8"

#define CLEAR_TO_END_OF_LINE "\033[K"

/// This codes are possibly not supported everywhere.
#define DISABLE_LINE_WRAPPING "\033[?7l"
#define ENABLE_LINE_WRAPPING "\033[?7h"


namespace DB
{
namespace ErrorCodes
{
extern const int POCO_EXCEPTION;
extern const int STD_EXCEPTION;
extern const int UNKNOWN_EXCEPTION;
extern const int NETWORK_ERROR;
extern const int NO_DATA_TO_INSERT;
extern const int BAD_ARGUMENTS;
extern const int UNKNOWN_PACKET_FROM_SERVER;
extern const int UNEXPECTED_PACKET_FROM_SERVER;
extern const int CLIENT_OUTPUT_FORMAT_SPECIFIED;
} // namespace ErrorCodes

inline char * readline(const char * prompt)
{
    std::string s;
    std::cout << prompt;
    std::getline(std::cin, s);

    if (!std::cin.good())
        return nullptr;
    return strdup(s.data());
}

class Client : public Poco::Util::Application
{
public:
    Client() = default;

private:
    using StringSet = std::unordered_set<String>;
    StringSet exit_strings{"exit",  "quit",    "logout", "учше", "йгше", "дщпщге", "exit;", "quit;", "logout;", "учшеж",
                           "йгшеж", "дщпщгеж", "q",      "й",    "\\q",  "\\Q",    "\\й",   "\\Й",   ":q",      "Жй"};
    bool is_interactive = true; /// Use either readline interface or batch mode.
    bool need_render_progress = true; /// Render query execution progress.
    bool echo_queries = false; /// Print queries before execution in batch mode.
    bool print_time_to_stderr = false; /// Output execution time to stderr in batch mode.
    bool stdin_is_not_tty = false; /// stdin is not a terminal.

    winsize terminal_size{}; /// Terminal size is needed to render progress bar.

    std::unique_ptr<Connection> connection; /// Connection to DB.
    String query_id; /// Current query_id.
    String query; /// Current query.

    String format; /// Query results output format.
    bool is_default_format = true; /// false, if format is set in the config or command line.
    size_t format_max_block_size = 0; /// Max block size for console output.
    String insert_format; /// Format of INSERT data that is read from stdin in batch mode.
    size_t insert_format_max_block_size = 0; /// Max block size when reading INSERT data.
    size_t max_client_network_bandwidth
        = 0; /// The maximum speed of data exchange over the network for the client in bytes per second.

    std::unique_ptr<Context> context = Context::createGlobal(Context::ApplicationType::CLIENT);

    /// Buffer that reads from stdin in batch mode.
    ReadBufferFromFileDescriptor std_in{STDIN_FILENO};

    /// Console output.
    WriteBufferFromFileDescriptor std_out{STDOUT_FILENO};
    std::unique_ptr<ShellCommand> pager_cmd;
    /// The user can specify to redirect query output to a file.
    std::optional<WriteBufferFromFile> out_file_buf;
    BlockOutputStreamPtr block_out_stream;

    String home_path;

    String current_profile;

    String prompt_by_server_display_name;

    /// How many rows have been read or written.
    size_t processed_rows = 0;

    /// Parsed query. Is used to determine some settings (e.g. format, output file).
    ASTPtr parsed_query;

    /// The last exception that was received from the server. Is used for the return code in batch mode.
    std::unique_ptr<Exception> last_exception;

    /// If the last query resulted in exception.
    bool got_exception = false;
    String server_version;
    String server_display_name;

    Stopwatch watch;

    /// The server periodically sends information about how much data was read since last time.
    Progress progress;
    bool show_progress_bar = false;

    size_t written_progress_chars = 0;
    bool written_first_block = false;

    struct ConnectionParameters
    {
        String host;
        UInt16 port = DBMS_DEFAULT_PORT;
        String default_database;
        String user;
        String password;
        Protocol::Compression compression = Protocol::Compression::Disable;
        ConnectionTimeouts timeouts;

        ConnectionParameters() = default;

        explicit ConnectionParameters(const Poco::Util::AbstractConfiguration & config)
        {
            host = config.getString("host", "localhost");
            port = config.getInt("port", config.getInt("tcp_port", DBMS_DEFAULT_PORT));

            default_database = config.getString("database", "");
            user = config.getString("user", "");
            password = config.getString("password", "");

            compression
                = config.getBool("compression", true) ? Protocol::Compression::Enable : Protocol::Compression::Disable;

            timeouts = ConnectionTimeouts(
                Poco::Timespan(config.getInt("connect_timeout", DBMS_DEFAULT_CONNECT_TIMEOUT_SEC), 0),
                Poco::Timespan(config.getInt("receive_timeout", DBMS_DEFAULT_RECEIVE_TIMEOUT_SEC), 0),
                Poco::Timespan(config.getInt("send_timeout", DBMS_DEFAULT_SEND_TIMEOUT_SEC), 0));
        }
    };

    ConnectionParameters connection_parameters;


    void initialize(Poco::Util::Application & self) override
    {
        Poco::Util::Application::initialize(self);

        const char * home_path_cstr = getenv("HOME");
        if (home_path_cstr)
            home_path = home_path_cstr;

        std::string config_path;
        if (config().has("config-file"))
            config_path = config().getString("config-file");

        if (!config_path.empty())
        {
            ConfigProcessor config_processor(config_path);
            auto loaded_config = config_processor.loadConfig();
            config().add(loaded_config.configuration);
        }

        /// settings and limits could be specified in config file, but passed settings has higher priority
#define EXTRACT_SETTING(TYPE, NAME, DEFAULT, DESCRIPTION)               \
    if (config().has(#NAME) && !context->getSettingsRef().NAME.changed) \
        context->setSetting(#NAME, config().getString(#NAME));
        APPLY_FOR_SETTINGS(EXTRACT_SETTING)
#undef EXTRACT_SETTING
    }


    int main(const std::vector<std::string> & /*args*/) override
    {
        try
        {
            return mainImpl();
        }
        catch (const Exception & e)
        {
            bool print_stack_trace = config().getBool("stacktrace", false);

            std::string text = e.displayText();

            /** If exception is received from server, then stack trace is embedded in message.
              * If exception is thrown on client, then stack trace is in separate field.
              */

            auto embedded_stack_trace_pos = text.find("Stack trace");
            if (std::string::npos != embedded_stack_trace_pos && !print_stack_trace)
                text.resize(embedded_stack_trace_pos);

            std::cerr << "Code: " << e.code() << ". " << text << std::endl << std::endl;

            /// Don't print the stack trace on the client if it was logged on the server.
            /// Also don't print the stack trace in case of network errors.
            if (print_stack_trace && e.code() != ErrorCodes::NETWORK_ERROR
                && std::string::npos == embedded_stack_trace_pos)
            {
                std::cerr << "Stack trace:" << std::endl << e.getStackTrace().toString();
            }

            /// If exception code isn't zero, we should return non-zero return code anyway.
            return e.code() ? e.code() : -1;
        }
        catch (const Poco::Exception & e)
        {
            std::cerr << "Poco::Exception: " << e.displayText() << std::endl;
            return ErrorCodes::POCO_EXCEPTION;
        }
        catch (const std::exception & e)
        {
            std::cerr << "std::exception: " << e.what() << std::endl;
            return ErrorCodes::STD_EXCEPTION;
        }
        catch (...)
        {
            std::cerr << "Unknown exception" << std::endl;
            return ErrorCodes::UNKNOWN_EXCEPTION;
        }
    }

    int mainImpl()
    {
        registerFunctions();
        registerAggregateFunctions();
        registerWindowFunctions();

        /// Batch mode is enabled if one of the following is true:
        /// - -e (--query) command line option is present.
        ///   The value of the option is used as the text of query (or of multiple queries).
        ///   If stdin is not a terminal, INSERT data for the first query is read from it.
        /// - stdin is not a terminal. In this case queries are read from it.
        stdin_is_not_tty = !isatty(STDIN_FILENO);
        if (stdin_is_not_tty || config().has("query"))
            is_interactive = false;

        std::cout << std::fixed << std::setprecision(3);
        std::cerr << std::fixed << std::setprecision(3);

        if (is_interactive)
            showClientVersion();

        is_default_format = !config().has("format");
        format = config().getString("format", is_interactive ? "PrettyCompactNoEscapes" : "TabSeparated");

        format_max_block_size = config().getInt("format_max_block_size", context->getSettingsRef().max_block_size);

        insert_format = "Values";
        insert_format_max_block_size
            = config().getInt("insert_format_max_block_size", context->getSettingsRef().max_insert_block_size);

        if (!is_interactive)
        {
            need_render_progress = config().getBool("progress", false);
            echo_queries = config().getBool("echo", false);
        }

        connection_parameters = ConnectionParameters(config());
        connect();

        /// Initialize DateLUT here to avoid counting time spent here as query execution time.
        DateLUT::instance();
        if (!context->getSettingsRef().use_client_time_zone)
        {
            const auto & time_zone = connection->getServerTimezone();
            if (!time_zone.empty())
            {
                try
                {
                    DateLUT::setDefaultTimezone(time_zone);
                }
                catch (...)
                {
                    std::cerr << "Warning: could not switch to server time zone: " << time_zone
                              << ", reason: " << getCurrentExceptionMessage(/* with_stacktrace = */ false) << std::endl
                              << "Proceeding with local time zone." << std::endl
                              << std::endl;
                }
            }
            else
            {
                std::cerr << "Warning: could not determine server time zone. "
                          << "Proceeding with local time zone." << std::endl
                          << std::endl;
            }
        }

        prompt_by_server_display_name
            = config().getRawString("prompt_by_server_display_name.default", "{display_name} :) ");

        /// Prompt may contain escape sequences including \e[ or \x1b[ sequences to set terminal color.
        {
            String unescaped_prompt_by_server_display_name;
            ReadBufferFromString in(prompt_by_server_display_name);
            readEscapedString(unescaped_prompt_by_server_display_name, in);
            prompt_by_server_display_name = std::move(unescaped_prompt_by_server_display_name);
        }

        /// Prompt may contain the following substitutions in a form of {name}.
        std::map<String, String> prompt_substitutions{
            {"host", connection_parameters.host},
            {"port", toString(connection_parameters.port)},
            {"user", connection_parameters.user},
            {"display_name", server_display_name},
        };

        /// Quite suboptimal.
        for (const auto & [key, value] : prompt_substitutions)
            boost::replace_all(prompt_by_server_display_name, "{" + key + "}", value);

        if (is_interactive)
        {
            if (!query_id.empty())
                throw Exception("query_id could be specified only in non-interactive mode", ErrorCodes::BAD_ARGUMENTS);
            if (print_time_to_stderr)
                throw Exception(
                    "time option could be specified only in non-interactive mode",
                    ErrorCodes::BAD_ARGUMENTS);

            loop();

            std::cout << "Bye." << std::endl;

            return 0;
        }
        else
        {
            query_id = config().getString("query_id", "");
            nonInteractive();

            if (last_exception)
                return last_exception->code();

            return 0;
        }
    }


    void connect()
    {
        if (is_interactive)
            std::cout << "Connecting to "
                      << (!connection_parameters.default_database.empty()
                              ? "database " + connection_parameters.default_database + " at "
                              : "")
                      << connection_parameters.host << ":" << connection_parameters.port
                      << (!connection_parameters.user.empty() ? " as user " + connection_parameters.user : "") << "."
                      << std::endl;

        connection = std::make_unique<Connection>(
            connection_parameters.host,
            connection_parameters.port,
            connection_parameters.default_database,
            connection_parameters.user,
            connection_parameters.password,
            connection_parameters.timeouts,
            "client",
            connection_parameters.compression);

        String server_name;
        UInt64 server_version_major = 0;
        UInt64 server_version_minor = 0;
        UInt64 server_version_patch = 0;
        if (max_client_network_bandwidth)
        {
            ThrottlerPtr throttler = std::make_shared<Throttler>(max_client_network_bandwidth, 0, "");
            connection->setThrottler(throttler);
        }

        connection->getServerVersion(server_name, server_version_major, server_version_minor, server_version_patch);
        server_version = fmt::format("{}.{}.{}", server_version_major, server_version_minor, server_version_patch);

        if (server_display_name = connection->getServerDisplayName(); server_display_name.length() == 0)
        {
            server_display_name = config().getString("host", "localhost");
        }

        if (is_interactive)
        {
            std::cout << "Connected to " << server_name << " server version " << server_version << "." << std::endl
                      << std::endl;
        }
    }


    /// Check if multi-line query is inserted from the paste buffer.
    /// Allows delaying the start of query execution until the entirety of query is inserted.
    static bool hasDataInSTDIN()
    {
        timeval timeout = {0, 0};
        fd_set fds; // NOLINT
        FD_ZERO(&fds);
        FD_SET(STDIN_FILENO, &fds); // NOLINT
        return select(1, &fds, nullptr, nullptr, &timeout) == 1;
    }

    inline String prompt() const
    {
        return boost::replace_all_copy(
            prompt_by_server_display_name,
            "{database}",
            config().getString("database", "default"));
    }

    void loop()
    {
        String query;
        String prev_query;

        while (char * line_read = readline(query.empty() ? prompt().c_str() : ":-] "))
        {
            String line = line_read;
            free(line_read); // NOLINT

            size_t ws = line.size();
            while (ws > 0 && isWhitespaceASCII(line[ws - 1]))
                --ws;

            if (ws == 0 || line.empty())
                continue;

            bool ends_with_semicolon = line[ws - 1] == ';';
            bool ends_with_backslash = line[ws - 1] == '\\';

            if (ends_with_backslash)
                line = line.substr(0, ws - 1);

            query += line;

            if (!ends_with_backslash && (ends_with_semicolon || (!config().has("multiline") && !hasDataInSTDIN())))
            {
                if (query != prev_query)
                {
                    /// Replace line breaks with spaces to prevent the following problem.
                    /// Every line of multi-line query is saved to history file as a separate line.
                    /// If the user restarts the client then after pressing the "up" button
                    /// every line of the query will be displayed separately.
                    std::string logged_query = query;
                    std::replace(logged_query.begin(), logged_query.end(), '\n', ' ');

                    prev_query = query;
                }

                try
                {
                    /// Determine the terminal size.
                    ioctl(0, TIOCGWINSZ, &terminal_size);

                    if (!process(query))
                        break;
                }
                catch (const Exception & e)
                {
                    std::cerr << std::endl
                              << "Exception on client:" << std::endl
                              << "Code: " << e.code() << ". " << e.displayText() << std::endl
                              << std::endl;

                    /// Client-side exception during query execution can result in the loss of
                    /// sync in the connection protocol.
                    /// So we reconnect and allow to enter the next query.
                    connect();
                }

                query = "";
            }
            else
            {
                query += '\n';
            }
        }
    }


    void nonInteractive()
    {
        String text;

        if (config().has("query"))
            text = config().getString("query");
        else
        {
            /// If 'query' parameter is not set, read a query from stdin.
            /// The query is read entirely into memory (streaming is disabled).
            ReadBufferFromFileDescriptor in(STDIN_FILENO);
            readStringUntilEOF(text, in);
        }

        process(text);
    }


    bool process(const String & text)
    {
        const auto ignore_error = config().getBool("ignore-error", false);
        if (config().has("multiquery"))
        {
            /// Several queries separated by ';'.
            /// INSERT data is ended by the end of line, not ';'.

            String query;

            const char * begin = text.data();
            const char * end = begin + text.size();

            while (begin < end)
            {
                const char * pos = begin;
                ASTPtr ast = parseQuery(pos, end, true);
                if (!ast)
                {
                    if (ignore_error)
                    {
                        Tokens tokens(begin, end);
                        TokenIterator token_iterator(tokens);
                        while (token_iterator->type != TokenType::Semicolon && token_iterator.isValid())
                            ++token_iterator;
                        begin = token_iterator->end;

                        continue;
                    }
                    return true;
                }

                auto * insert = typeid_cast<ASTInsertQuery *>(&*ast);

                if (insert && insert->data)
                {
                    pos = find_first_symbols<'\n'>(insert->data, end);
                    insert->end = pos;
                }

                query = text.substr(begin - text.data(), pos - begin);

                begin = pos;
                while (isWhitespaceASCII(*begin) || *begin == ';')
                    ++begin;

                try
                {
                    if (!processSingleQuery(query, ast) && !ignore_error)
                        return false;
                }
                catch (...)
                {
                    std::cerr << "Error on processing query: " << query << std::endl
                              << getCurrentExceptionMessage(true);
                    got_exception = true;
                }

                if (got_exception && !ignore_error)
                {
                    if (is_interactive)
                        break;
                    else
                        return false;
                }
            }

            return true;
        }
        else
        {
            return processSingleQuery(text);
        }
    }


    bool processSingleQuery(const String & line, ASTPtr parsed_query_ = nullptr)
    {
        if (exit_strings.end() != exit_strings.find(line))
            return false;

        resetOutput();
        got_exception = false;

        if (echo_queries)
        {
            writeString(line, std_out);
            writeChar('\n', std_out);
            std_out.next();
        }

        watch.restart();

        query = line;

        /// Some parts of a query (result output and formatting) are executed client-side.
        /// Thus we need to parse the query.
        parsed_query = parsed_query_;

        if (!parsed_query)
        {
            const char * begin = query.data();
            parsed_query = parseQuery(begin, begin + query.size(), false);
        }

        if (!parsed_query)
            return true;

        processed_rows = 0;
        progress.reset();
        show_progress_bar = false;
        written_progress_chars = 0;
        written_first_block = false;

        const auto * set_query = typeid_cast<const ASTSetQuery *>(&*parsed_query);
        const auto * use_query = typeid_cast<const ASTUseQuery *>(&*parsed_query);
        /// INSERT query for which data transfer is needed (not an INSERT SELECT) is processed separately.
        const auto * insert = typeid_cast<const ASTInsertQuery *>(&*parsed_query);

        connection->forceConnected();

        if (insert && !insert->select)
            processInsertQuery();
        else
            processOrdinaryQuery();

        /// Do not change context (current DB, settings) in case of an exception.
        if (!got_exception)
        {
            if (set_query)
            {
                /// Save all changes in settings to avoid losing them if the connection is lost.
                for (const auto & change : set_query->changes)
                {
                    if (change.name == "profile")
                        current_profile = change.value.safeGet<String>();
                    else
                        context->setSetting(change.name, change.value);
                }
            }

            if (use_query)
            {
                const String & new_database = use_query->database;
                /// If the client initiates the reconnection, it takes the settings from the config.
                config().setString("database", new_database);
                /// If the connection initiates the reconnection, it uses its variable.
                connection->setDefaultDatabase(new_database);
            }
        }

        if (is_interactive)
        {
            std::cout << std::endl << processed_rows << " rows in set. Elapsed: " << watch.elapsedSeconds() << " sec. ";

            if (progress.rows >= 1000)
                writeFinalProgress();

            std::cout << std::endl << std::endl;
        }
        else if (print_time_to_stderr)
        {
            std::cerr << watch.elapsedSeconds() << "\n";
        }

        return true;
    }

    /// Process the query that doesn't require transfering data blocks to the server.
    void processOrdinaryQuery()
    {
        connection->sendQuery(query, query_id, QueryProcessingStage::Complete, &context->getSettingsRef(), nullptr);
        receiveResult();
    }


    /// Process the query that requires transfering data blocks to the server.
    void processInsertQuery()
    {
        /// Send part of query without data, because data will be sent separately.
        const ASTInsertQuery & parsed_insert_query = typeid_cast<const ASTInsertQuery &>(*parsed_query);
        String query_without_data
            = parsed_insert_query.data ? query.substr(0, parsed_insert_query.data - query.data()) : query;

        if (!parsed_insert_query.data && (is_interactive || (stdin_is_not_tty && std_in.eof())))
            throw Exception("No data to insert", ErrorCodes::NO_DATA_TO_INSERT);

        connection->sendQuery(
            query_without_data,
            query_id,
            QueryProcessingStage::Complete,
            &context->getSettingsRef(),
            nullptr);

        /// Receive description of table structure.
        Block sample;
        if (receiveSampleBlock(sample))
        {
            /// If structure was received (thus, server has not thrown an exception),
            /// send our data with that structure.
            sendData(sample);
            receivePacket();
        }
    }


    ASTPtr parseQuery(const char *& pos, const char * end, bool allow_multi_statements)
    {
        ParserQuery parser(end);
        ASTPtr res;

        const auto ignore_error = config().getBool("ignore-error", false);

        if (is_interactive || ignore_error)
        {
            String message;
            res = tryParseQuery(parser, pos, end, message, true, "", allow_multi_statements, 0);

            if (!res)
            {
                std::cerr << std::endl << message << std::endl << std::endl;
                return nullptr;
            }
        }
        else
            res = parseQueryAndMovePosition(parser, pos, end, "", allow_multi_statements, 0);

        if (is_interactive)
        {
            std::cout << std::endl;
            formatAST(*res, std::cout);
            std::cout << std::endl << std::endl;
        }

        return res;
    }


    void sendData(Block & sample)
    {
        /// If INSERT data must be sent.
        const auto * parsed_insert_query = typeid_cast<const ASTInsertQuery *>(&*parsed_query);
        if (!parsed_insert_query)
            return;

        if (parsed_insert_query->data)
        {
            /// Send data contained in the query.
            ReadBufferFromMemory data_in(
                parsed_insert_query->data,
                parsed_insert_query->end - parsed_insert_query->data);
            sendDataFrom(data_in, sample);
        }
        else if (!is_interactive)
        {
            /// Send data read from stdin.
            sendDataFrom(std_in, sample);
        }
        else
            throw Exception("No data to insert", ErrorCodes::NO_DATA_TO_INSERT);
    }


    void sendDataFrom(ReadBuffer & buf, Block & sample)
    {
        String current_format = insert_format;

        /// Data format can be specified in the INSERT query.
        if (auto * insert = typeid_cast<ASTInsertQuery *>(&*parsed_query))
            if (!insert->format.empty())
                current_format = insert->format;

        BlockInputStreamPtr block_input
            = context->getInputFormat(current_format, buf, sample, insert_format_max_block_size);

        BlockInputStreamPtr async_block_input = std::make_shared<AsynchronousBlockInputStream>(block_input);

        async_block_input->readPrefix();

        while (true)
        {
            Block block = async_block_input->read();
            connection->sendData(block);
            processed_rows += block.rows();

            if (!block)
                break;
        }

        async_block_input->readSuffix();
    }


    /// Flush all buffers.
    void resetOutput()
    {
        block_out_stream = nullptr;
        if (pager_cmd)
        {
            pager_cmd->in.close();
            pager_cmd->wait();
        }
        pager_cmd = nullptr;
        if (out_file_buf)
        {
            out_file_buf->next();
            out_file_buf.reset();
        }
        std_out.next();
    }


    /// Receives and processes packets coming from server.
    /// Also checks if query execution should be cancelled.
    void receiveResult()
    {
        InterruptListener interrupt_listener;
        bool cancelled = false;

        while (true)
        {
            /// Has the Ctrl+C been pressed and thus the query should be cancelled?
            /// If this is the case, inform the server about it and receive the remaining packets
            /// to avoid losing sync.
            if (!cancelled)
            {
                if (interrupt_listener.check())
                {
                    connection->sendCancel();
                    cancelled = true;
                    if (is_interactive)
                        std::cout << "Cancelling query." << std::endl;

                    /// Pressing Ctrl+C twice results in shut down.
                    interrupt_listener.unblock();
                }
                else if (!connection->poll(1000000))
                    continue; /// If there is no new data, continue checking whether the query was cancelled after a timeout.
            }

            if (!receivePacket())
                break;
        }

        if (cancelled && is_interactive)
            std::cout << "Query was cancelled." << std::endl;
    }


    /// Receive a part of the result, or progress info or an exception and process it.
    /// Returns true if one should continue receiving packets.
    bool receivePacket()
    {
        Connection::Packet packet = connection->receivePacket();

        switch (packet.type)
        {
        case Protocol::Server::Data:
            onData(packet.block);
            return true;

        case Protocol::Server::Progress:
            onProgress(packet.progress);
            return true;

        case Protocol::Server::ProfileInfo:
            onProfileInfo(packet.profile_info);
            return true;

        case Protocol::Server::Extremes:
            onExtremes(packet.block);
            return true;

        case Protocol::Server::Exception:
            onException(*packet.exception);
            last_exception = std::move(packet.exception);
            return false;

        case Protocol::Server::EndOfStream:
            onEndOfStream();
            return false;

        default:
            throw Exception("Unknown packet from server", ErrorCodes::UNKNOWN_PACKET_FROM_SERVER);
        }
    }


    /// Receive the block that serves as an example of the structure of table where data will be inserted.
    bool receiveSampleBlock(Block & out)
    {
        Connection::Packet packet = connection->receivePacket();

        switch (packet.type)
        {
        case Protocol::Server::Data:
            out = packet.block;
            return true;

        case Protocol::Server::Exception:
            onException(*packet.exception);
            last_exception = std::move(packet.exception);
            return false;

        default:
            throw NetException(
                "Unexpected packet from server (expected Data, got " + String(Protocol::Server::toString(packet.type))
                    + ")",
                ErrorCodes::UNEXPECTED_PACKET_FROM_SERVER);
        }
    }


    void initBlockOutputStream(const Block & block)
    {
        if (!block_out_stream)
        {
            WriteBuffer * out_buf = nullptr;
            String pager = config().getString("pager", "");
            if (!pager.empty())
            {
                signal(SIGPIPE, SIG_IGN); // NOLINT
                pager_cmd = ShellCommand::execute(pager, true);
                out_buf = &pager_cmd->in;
            }
            else
            {
                out_buf = &std_out;
            }

            String current_format = format;

            /// The query can specify output format or output file.
            if (auto * query_with_output = dynamic_cast<ASTQueryWithOutput *>(&*parsed_query))
            {
                if (query_with_output->out_file != nullptr)
                {
                    const auto & out_file_node = typeid_cast<const ASTLiteral &>(*query_with_output->out_file);
                    const auto & out_file = out_file_node.value.safeGet<std::string>();
                    out_file_buf.emplace(out_file, DBMS_DEFAULT_BUFFER_SIZE, O_WRONLY | O_EXCL | O_CREAT);
                    out_buf = &*out_file_buf;

                    // We are writing to file, so default format is the same as in non-interactive mode.
                    if (is_interactive && is_default_format)
                        current_format = "TabSeparated";
                }
                if (query_with_output->format != nullptr)
                {
                    const auto & id = typeid_cast<const ASTIdentifier &>(*query_with_output->format);
                    current_format = id.name;
                }
            }

            block_out_stream = context->getOutputFormat(current_format, *out_buf, block);
            block_out_stream->writePrefix();
        }
    }


    void onData(Block & block)
    {
        if (written_progress_chars)
            clearProgress();

        if (!block)
            return;

        processed_rows += block.rows();
        initBlockOutputStream(block);

        /// The header block containing zero rows was used to initialize block_out_stream, do not output it.
        if (block.rows() != 0)
        {
            block_out_stream->write(block);
            written_first_block = true;
        }

        /// Received data block is immediately displayed to the user.
        block_out_stream->flush();
    }


    void onExtremes(Block & block)
    {
        initBlockOutputStream(block);
        block_out_stream->setExtremes(block);
    }


    void onProgress(const Progress & value)
    {
        progress.incrementPiecewiseAtomically(value);
        if (block_out_stream)
            block_out_stream->onProgress(value);
        writeProgress();
    }


    void clearProgress()
    {
        std::cerr << RESTORE_CURSOR_POSITION CLEAR_TO_END_OF_LINE;
        written_progress_chars = 0;
    }


    void writeProgress()
    {
        if (!need_render_progress)
            return;

        static size_t increment = 0;
        static const char * indicators[8] = {
            "\033[1;30m→\033[0m",
            "\033[1;31m↘\033[0m",
            "\033[1;32m↓\033[0m",
            "\033[1;33m↙\033[0m",
            "\033[1;34m←\033[0m",
            "\033[1;35m↖\033[0m",
            "\033[1;36m↑\033[0m",
            "\033[1m↗\033[0m",
        };

        if (written_progress_chars)
            clearProgress();
        else
            std::cerr << SAVE_CURSOR_POSITION;

        std::stringstream message;
        message << indicators[increment % 8] << std::fixed << std::setprecision(3) << " Progress: ";

        message << formatReadableQuantity(progress.rows) << " rows, "
                << formatReadableSizeWithDecimalSuffix(progress.bytes);

        size_t elapsed_ns = watch.elapsed();
        if (elapsed_ns)
            message << " (" << formatReadableQuantity(progress.rows * 1000000000.0 / elapsed_ns) << " rows/s., "
                    << formatReadableSizeWithDecimalSuffix(progress.bytes * 1000000000.0 / elapsed_ns) << "/s.) ";
        else
            message << ". ";

        written_progress_chars = message.str().size() - (increment % 8 == 7 ? 10 : 13);
        std::cerr << DISABLE_LINE_WRAPPING << message.rdbuf();

        /// If the approximate number of rows to process is known, we can display a progress bar and percentage.
        if (progress.total_rows > 0)
        {
            size_t total_rows_corrected = std::max(progress.rows, progress.total_rows);

            /// To avoid flicker, display progress bar only if .5 seconds have passed since query execution start
            ///  and the query is less than halfway done.

            if (elapsed_ns > 500000000)
            {
                /// Trigger to start displaying progress bar. If query is mostly done, don't display it.
                if (progress.rows * 2 < total_rows_corrected)
                    show_progress_bar = true;

                if (show_progress_bar)
                {
                    ssize_t width_of_progress_bar
                        = static_cast<ssize_t>(terminal_size.ws_col) - written_progress_chars - strlen(" 99%");
                    if (width_of_progress_bar > 0)
                    {
                        std::string bar = UnicodeBar::render(
                            UnicodeBar::getWidth(progress.rows, 0, total_rows_corrected, width_of_progress_bar));
                        std::cerr << "\033[0;32m" << bar << "\033[0m";
                        if (width_of_progress_bar > static_cast<ssize_t>(bar.size() / UNICODE_BAR_CHAR_SIZE))
                            std::cerr << std::string(width_of_progress_bar - bar.size() / UNICODE_BAR_CHAR_SIZE, ' ');
                    }
                }
            }

            /// Underestimate percentage a bit to avoid displaying 100%.
            std::cerr << ' ' << (99 * progress.rows / total_rows_corrected) << '%';
        }

        std::cerr << ENABLE_LINE_WRAPPING;
        ++increment;
    }


    void writeFinalProgress()
    {
        std::cout << "Processed " << formatReadableQuantity(progress.rows) << " rows, "
                  << formatReadableSizeWithDecimalSuffix(progress.bytes);

        size_t elapsed_ns = watch.elapsed();
        if (elapsed_ns)
            std::cout << " (" << formatReadableQuantity(progress.rows * 1000000000.0 / elapsed_ns) << " rows/s., "
                      << formatReadableSizeWithDecimalSuffix(progress.bytes * 1000000000.0 / elapsed_ns) << "/s.) ";
        else
            std::cout << ". ";
    }


    void onException(const Exception & e)
    {
        resetOutput();
        got_exception = true;

        std::string text = e.displayText();

        auto embedded_stack_trace_pos = text.find("Stack trace");
        if (std::string::npos != embedded_stack_trace_pos && !config().getBool("stacktrace", false))
            text.resize(embedded_stack_trace_pos);

        std::cerr << "Received exception from server (version " << server_version << "):" << std::endl
                  << "Code: " << e.code() << ". " << text << std::endl;
    }


    void onProfileInfo(const BlockStreamProfileInfo & profile_info)
    {
        if (profile_info.hasAppliedLimit() && block_out_stream)
            block_out_stream->setRowsBeforeLimit(profile_info.getRowsBeforeLimit());
    }


    void onEndOfStream()
    {
        if (block_out_stream)
            block_out_stream->writeSuffix();

        resetOutput();

        if (is_interactive && !written_first_block)
            std::cout << "Ok." << std::endl;
    }

    static void showClientVersion()
    {
        std::cout << "TiFlash client version " << TiFlashBuildInfo::getReleaseVersion() << "." << std::endl;
    }

public:
    void init(int argc, char ** argv)
    {
        /// Don't parse options with Poco library. We need more sophisticated processing.
        stopOptionsProcessing();

        /** We allow different groups of arguments:
          * - common arguments;
          * Split these groups before processing.
          */
        using Arguments = std::vector<const char *>;

        Arguments common_arguments{""}; /// 0th argument is ignored.

        for (int arg_num = 1; arg_num < argc; ++arg_num)
        {
            const char * arg = argv[arg_num];
            common_arguments.emplace_back(arg);
        }

#define DECLARE_SETTING(TYPE, NAME, DEFAULT, DESCRIPTION) \
    (#NAME, boost::program_options::value<std::string>(), DESCRIPTION)

        /// Main commandline options related to client functionality and all parameters from Settings.
        boost::program_options::options_description main_description("Main options");
        // clang-format off
        main_description.add_options()
            ("help", "produce help message")
            ("config-file,c", boost::program_options::value<std::string>(), "config-file path")
            ("host,h", boost::program_options::value<std::string>()->default_value("localhost"), "server host")
            ("port", boost::program_options::value<int>()->default_value(9000), "server port")
            ("user,u", boost::program_options::value<std::string>(), "user")
            ("password", boost::program_options::value<std::string>(), "password")
            ("query_id", boost::program_options::value<std::string>(), "query_id")
            ("query,q", boost::program_options::value<std::string>(), "query")
            ("database,d", boost::program_options::value<std::string>(), "database")
            ("pager", boost::program_options::value<std::string>(), "pager")
            ("multiline,m", "multiline")
            ("multiquery,n", "multiquery")
            ("ignore-error", "Do not stop processing in multiquery mode")
            ("format,f", boost::program_options::value<std::string>(), "default output format")
            ("time,t", "print query execution time to stderr in non-interactive mode (for benchmarks)")
            ("stacktrace", "print stack traces of exceptions")
            ("progress", "print progress even in non-interactive mode")
            ("version,V", "print version information and exit")
            ("echo", "in batch mode, print query before execution")
            ("max_client_network_bandwidth", boost::program_options::value<int>(), "the maximum speed of data exchange over the network for the client in bytes per second.")
            ("compression", boost::program_options::value<bool>(), "enable or disable compression")
            APPLY_FOR_SETTINGS(DECLARE_SETTING)
        ;
        // clang-format on
#undef DECLARE_SETTING

        /// Parse main commandline options.
        boost::program_options::parsed_options parsed
            = boost::program_options::command_line_parser(common_arguments.size(), common_arguments.data())
                  .options(main_description)
                  .run();
        boost::program_options::variables_map options;
        boost::program_options::store(parsed, options);

        if (options.count("version") || options.count("V"))
        {
            showClientVersion();
            exit(0);
        }

        /// Output of help message.
        if (options.count("help")
            || (options.count("host")
                && options["host"].as<std::string>() == "elp")) /// If user writes -help instead of --help.
        {
            std::cout << main_description << "\n";
            exit(0);
        }

        /// Extract settings and limits from the options.
#define EXTRACT_SETTING(TYPE, NAME, DEFAULT, DESCRIPTION) \
    if (options.count(#NAME))                             \
        context->setSetting(#NAME, options[#NAME].as<std::string>());
        APPLY_FOR_SETTINGS(EXTRACT_SETTING)
#undef EXTRACT_SETTING

        /// Save received data into the internal config.
        if (options.count("config-file"))
            config().setString("config-file", options["config-file"].as<std::string>());
        if (options.count("host") && !options["host"].defaulted())
            config().setString("host", options["host"].as<std::string>());
        if (options.count("query_id"))
            config().setString("query_id", options["query_id"].as<std::string>());
        if (options.count("query"))
            config().setString("query", options["query"].as<std::string>());
        if (options.count("database"))
            config().setString("database", options["database"].as<std::string>());
        if (options.count("pager"))
            config().setString("pager", options["pager"].as<std::string>());

        if (options.count("port") && !options["port"].defaulted())
            config().setInt("port", options["port"].as<int>());
        if (options.count("user"))
            config().setString("user", options["user"].as<std::string>());
        if (options.count("password"))
            config().setString("password", options["password"].as<std::string>());

        if (options.count("multiline"))
            config().setBool("multiline", true);
        if (options.count("multiquery"))
            config().setBool("multiquery", true);
        if (options.count("ignore-error"))
            config().setBool("ignore-error", true);
        if (options.count("format"))
            config().setString("format", options["format"].as<std::string>());
        if (options.count("stacktrace"))
            config().setBool("stacktrace", true);
        if (options.count("progress"))
            config().setBool("progress", true);
        if (options.count("echo"))
            config().setBool("echo", true);
        if (options.count("time"))
            print_time_to_stderr = true;
        if (options.count("max_client_network_bandwidth"))
            max_client_network_bandwidth = options["max_client_network_bandwidth"].as<int>();
        if (options.count("compression"))
            config().setBool("compression", options["compression"].as<bool>());
    }
};

} // namespace DB


int mainEntryClickHouseClient(int argc, char ** argv)
{
    DB::Client client;

    try
    {
        client.init(argc, argv);
    }
    catch (const boost::program_options::error & e)
    {
        std::cerr << "Bad arguments: " << e.what() << std::endl;
        return 1;
    }

    return client.run();
}
