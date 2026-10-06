##############################################################
# POPULATION-ENGINE WARMUP                                   #
##############################################################

"""
    warmup_programs_population!(cache, tpg, program_ids, samples, si, ml, ma)

Fill `cache` with the bid of every program in `program_ids` on every sample.
`samples` iterates `(x, y, hashed_x)` tuples.

Programs that already hold a value for every sample are skipped. The remaining
ones are compiled and merged into one `UTCGP.PopulationSequentialProgram`, so
computation shared between programs runs once per sample; samples are
evaluated in parallel by the threads.

Returns the number of programs evaluated.
"""
function warmup_programs_population!(
        cache::TPGEvaluationCache, tpg::TangledProgramGraph, program_ids,
        samples, si::SharedInput, ml::MetaLibrary, ma::modelArchitecture
    )
    samples = collect(samples)
    hashes = UInt64[s[3] for s in samples]
    todo = ProgramID[]
    for pid in program_ids
        create_key_in_cache_or_nothing!(cache, pid)
        subcache = cache.program_caches[pid]
        if any(h -> isnothing(get(subcache, h, nothing)), hashes)
            push!(todo, pid)
        end
    end
    isempty(todo) && return 0

    @timeit_debug to "Population warmup : compile" begin
        sequential_programs = UTCGP.SequentialProgram[]
        for pid in todo
            genome = tpg.programs[pid].genome
            program = UTCGP.decode_with_output_nodes(genome, ml, ma, si).programs[1]
            push!(sequential_programs, UTCGP.compile_program(program, ma, ml; safe = true))
        end
        population_program = UTCGP.PopulationSequentialProgram(sequential_programs)
    end

    sample_inputs = Tuple[Tuple(s[1]) for s in samples]
    @timeit_debug to "Population warmup : eval" outputs =
        UTCGP.evaluate_population_sequential_program_on_samples(population_program, sample_inputs)

    for (j, h) in enumerate(hashes), (i, pid) in enumerate(todo)
        set_cached_value!(cache, pid, h, outputs[j][i][1]) # one output per program
    end
    return length(todo)
end

"""
    warmup_programs_individual!(cache, tpg, program_ids, samples, si, ml, ma)

Classic warmup: evaluate each program alone, threads over samples.
"""
function warmup_programs_individual!(
        cache::TPGEvaluationCache, tpg::TangledProgramGraph, program_ids,
        samples, si::SharedInput, ml::MetaLibrary, ma::modelArchitecture
    )
    samples = collect(samples)
    for pid in program_ids
        tpg_program = find_program_by_id(tpg, pid)
        create_key_in_cache_or_nothing!(cache, pid)
        subcache = cache.program_caches[pid]
        program_for_threads = [deepcopy(tpg_program) for i in 1:Threads.nthreads()]
        Threads.@threads :static for (x, y, hashed_x) in samples
            thread_prog = program_for_threads[Threads.threadid()]
            if get(subcache, hashed_x, nothing) |> isnothing
                evaluate(thread_prog, x, hashed_x, cache, si, ml, ma)
            end
        end
    end
    return length(program_ids)
end

"""
    warmup_programs!(engine, cache, tpg, program_ids, samples, si, ml, ma)

Dispatch on `engine`: `:population` or `:individual`.
"""
function warmup_programs!(engine::Symbol, args...)
    engine === :population && return warmup_programs_population!(args...)
    engine === :individual && return warmup_programs_individual!(args...)
    return error("Unknown TPG engine $engine. Use :individual or :population.")
end
