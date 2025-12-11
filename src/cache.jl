using ThreadSafeDicts

##############################################################
# CATCHING MECHANISM                                         #
##############################################################
@enum CacheMode NoCache PerInputCache LRUCacheMode
# TODO make this modes parametric structs to hold the return type of the programs.
# it will be better than making the current dict{hash, any} to dict{hash, T}

"""
    TPGEvaluationCache
A struct to manage the global evaluation cache for programs within a TangledProgramGraph.
"""
mutable struct TPGEvaluationCache
    program_caches::ThreadSafeDict{ProgramID, Any} # program_id => Dict{input_hash, output_value} or LRU{input_hash, output_value}
    mode::CacheMode
    enabled::Bool
    lru_max_size::Int # Max size for LRU cache

    function TPGEvaluationCache(mode::CacheMode = PerInputCache; lru_max_size::Int = 1000)
        return new(Dict{ProgramID, Any}(), mode, true, lru_max_size)
    end
end

function init_cache(mode::CacheMode = PerInputCache; lru_max_size::Int = 1000)::TPGEvaluationCache
    cache = TPGEvaluationCache(mode, lru_max_size = lru_max_size)
    if mode == NoCache
        disable_cache!(cache)
    end
    return cache
end

function enable_cache!(cache::TPGEvaluationCache)
    return cache.enabled = true
end

function disable_cache!(cache::TPGEvaluationCache)
    return cache.enabled = false
end

function is_cache_enabled(cache::TPGEvaluationCache)::Bool
    return cache.enabled
end

function clear_cache!(cache::TPGEvaluationCache)
    for (program_id, program_cache) in cache.program_caches
        empty!(program_cache)
    end
    return empty!(cache.program_caches)
end

function get_cached_value(cache::TPGEvaluationCache, program_id::ProgramID, input_hash::UInt64)
    if is_cache_enabled(cache)
        if haskey(cache.program_caches, program_id)
            program_cache = cache.program_caches[program_id]
            return get(program_cache, input_hash, nothing)
        end
    end
    return nothing
end

function create_key_in_cache_or_nothing!(cache::TPGEvaluationCache, key::ProgramID)
    if !is_cache_enabled(cache)
        return
    end
    if !haskey(cache.program_caches, key)
        if cache.mode == LRUCacheMode
            cache.program_caches[key] = LRU{UInt64, Any}(maxsize = cache.lru_max_size)
        else # PerInputCache
            cache.program_caches[key] = Dict{UInt64, Any}()
        end
    end
    return
end

function set_cached_value!(cache::TPGEvaluationCache, program_id::ProgramID, input_hash::UInt64, value::Any)
    if is_cache_enabled(cache)
        create_key_in_cache_or_nothing!(cache, program_id)
        program_cache = cache.program_caches[program_id]
        program_cache[input_hash] = value
    end
    return value
end

function copy_cache!(cache::TPGEvaluationCache, from_program_id::ProgramID, to_program_id::ProgramID)
    return if is_cache_enabled(cache)
        if haskey(cache.program_caches, from_program_id)
            create_key_in_cache_or_nothing!(cache, to_program_id)
            to_cache = cache.program_caches[to_program_id]
            for (k, v) in cache.program_caches[from_program_id]
                to_cache[k] = v
            end
        end
    end
end

function programs_without_cache(cache::TPGEvaluationCache, tpg::TangledProgramGraph)
    ps_in_tpg = collect(keys(tpg.programs))
    ps_in_cache = collect(keys(cache.program_caches))
    ps_without_cache = setdiff(ps_in_tpg, ps_in_cache)
    print("Programs without cache : $(length(ps_without_cache))")
    return ps_without_cache
end
