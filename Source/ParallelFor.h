#pragma once

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#include "Assert.h"

namespace FilterFitter
{
    // Worker count a call with an unbounded item count would use, i.e. the machine's parallelism. Reported by the benchmark; never less than 1.
    inline uint32_t GetHardwareWorkerCount()
    {
        uint32_t const hardware = std::thread::hardware_concurrency();

        return ( hardware > 0 ) ? hardware : 1;
    }

    // Ranges a call will use: min( numItems, hardware concurrency ), and never more than the item count. Callers that keep one scratch buffer per worker size their array from this.
    inline uint32_t GetWorkerCount( uint32_t numItems )
    {
        if ( numItems == 0 )
        {
            return 0;
        }

        uint32_t const maxWorkers = GetHardwareWorkerCount();

        return ( numItems < maxWorkers ) ? numItems : maxWorkers;
    }

    //-------------------------------------------------------------------------

    namespace Detail
    {
        // True on a pool worker thread, so a nested call can fall back to serial instead of deadlocking on a pool it is already occupying.
        inline thread_local bool t_isPoolWorker = false;

        //---------------------------------------------------------------------

        inline uint32_t RangeBegin( uint32_t numItems, uint32_t numRanges, uint32_t rangeIndex )
        {
            // Widened because numItems can exceed 2^31 / numRanges
            return static_cast<uint32_t>( ( static_cast<uint64_t>( numItems ) * rangeIndex ) / numRanges );
        }

        //---------------------------------------------------------------------

        class ThreadPool
        {
        public:

            using Body = void ( * )( void* pContext, uint32_t begin, uint32_t end, uint32_t workerIndex );

        public:

            explicit ThreadPool( uint32_t numWorkers )
                : m_numWorkers( numWorkers )
            {
                m_workers.reserve( numWorkers );

                for ( uint32_t workerIndex = 1; workerIndex <= numWorkers; ++workerIndex )
                {
                    m_workers.emplace_back( [this, workerIndex] () { WorkerLoop( workerIndex ); } );
                }

                // Do not let a call race the threads' own startup
                std::unique_lock<std::mutex> lock( m_mutex );
                m_started.wait( lock, [this] () { return m_workersStarted == m_numWorkers; } );
            }

            ~ThreadPool()
            {
                {
                    std::lock_guard<std::mutex> lock( m_mutex );
                    m_shutdown = true;
                    ++m_generation;
                }

                m_workAvailable.notify_all();

                for ( std::thread& worker : m_workers )
                {
                    if ( worker.joinable() )
                    {
                        worker.join();
                    }
                }
            }

            ThreadPool( ThreadPool const& ) = delete;
            ThreadPool& operator=( ThreadPool const& ) = delete;

            //-----------------------------------------------------------------

            void RunRanges( uint32_t numItems, uint32_t numRanges, Body body, void* pContext )
            {
                if ( numRanges <= 1 )
                {
                    body( pContext, 0, numItems, 0 );
                    return;
                }

                {
                    std::lock_guard<std::mutex> lock( m_mutex );

                    m_pBody = body;
                    m_pContext = pContext;
                    m_numItems = numItems;
                    m_numRanges = numRanges;

                    // Every pool worker reports, including those whose index is past this call's range count and therefore do no work.
                    // That keeps the completion condition independent of numRanges.
                    m_workersRemaining = m_numWorkers;

                    ++m_generation;
                }

                m_workAvailable.notify_all();

                // The calling thread takes range 0
                body( pContext, RangeBegin( numItems, numRanges, 0 ), RangeBegin( numItems, numRanges, 1 ), 0 );

                std::unique_lock<std::mutex> lock( m_mutex );
                m_workComplete.wait( lock, [this] () { return m_workersRemaining == 0; } );
            }

        private:

            void WorkerLoop( uint32_t workerIndex )
            {
                t_isPoolWorker = true;

                {
                    std::lock_guard<std::mutex> lock( m_mutex );
                    ++m_workersStarted;
                }

                m_started.notify_all();

                uint64_t seenGeneration = 0;

                for ( ;; )
                {
                    Body     body = nullptr;
                    void*    pContext = nullptr;
                    uint32_t numItems = 0;
                    uint32_t numRanges = 0;

                    {
                        std::unique_lock<std::mutex> lock( m_mutex );

                        m_workAvailable.wait( lock, [this, &seenGeneration] () { return m_generation != seenGeneration; } );
                        seenGeneration = m_generation;

                        if ( m_shutdown )
                        {
                            return;
                        }

                        body = m_pBody;
                        pContext = m_pContext;
                        numItems = m_numItems;
                        numRanges = m_numRanges;
                    }

                    // A parked worker cannot be missed: the caller does not return until every worker has decremented, so all workers are back in wait() before the next generation can be set.
                    if ( workerIndex < numRanges )
                    {
                        body( pContext, RangeBegin( numItems, numRanges, workerIndex ), RangeBegin( numItems, numRanges, workerIndex + 1 ), workerIndex );
                    }

                    {
                        std::lock_guard<std::mutex> lock( m_mutex );

                        if ( --m_workersRemaining == 0 )
                        {
                            m_workComplete.notify_all();
                        }
                    }
                }
            }

        private:

            std::vector<std::thread>    m_workers;
            uint32_t                    m_numWorkers = 0;

            std::mutex                  m_mutex;
            std::condition_variable     m_started;
            std::condition_variable     m_workAvailable;
            std::condition_variable     m_workComplete;

            Body                        m_pBody = nullptr;
            void*                       m_pContext = nullptr;
            uint32_t                    m_numItems = 0;
            uint32_t                    m_numRanges = 0;
            uint32_t                    m_workersRemaining = 0;
            uint32_t                    m_workersStarted = 0;
            uint64_t                    m_generation = 0;
            bool                        m_shutdown = false;
        };

        //---------------------------------------------------------------------

        inline ThreadPool& GetThreadPool()
        {
            // One pool for the process, sized so that the pool plus the calling thread is the machine's parallelism
            static ThreadPool pool( ( GetHardwareWorkerCount() > 1 ) ? ( GetHardwareWorkerCount() - 1 ) : 1 );

            return pool;
        }
    }

    //-------------------------------------------------------------------------

    // body( begin, end, workerIndex ) evaluates the half-open range [begin, end) for the worker index, which is in [0, GetWorkerCount( numItems )). 
    // Each worker index maps to the same range on every call, so per-worker scratch indexed by it needs no locking.
    template< typename TBody >
    void ParallelForRanges( uint32_t numItems, TBody body )
    {
        uint32_t const numWorkers = GetWorkerCount( numItems );

        if ( numWorkers == 0 )
        {
            return;
        }

        if ( numWorkers == 1 || Detail::t_isPoolWorker )
        {
            body( 0, numItems, 0 );
            return;
        }

        // Captureless, so it converts to the function pointer the pool stores.
        // The body itself is passed as the context and outlives the call because every lane is joined before this returns.
        Detail::GetThreadPool().RunRanges( numItems, numWorkers,
            [] ( void* pContext, uint32_t begin, uint32_t end, uint32_t workerIndex )
        {
            ( *static_cast<TBody*>( pContext ) )( begin, end, workerIndex );
        },
            &body );
    }
}
