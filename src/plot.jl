##############################################################
# PLOTTING                                                   #
##############################################################

"""
    describe_program(tpg_program, ml, ma, si)::String

The function names the program runs, in execution order, joined by " → ".
"""
function describe_program(tpg_program::TPGProgram, ml::MetaLibrary, ma::modelArchitecture, si::SharedInput)::String
    program = UTCGP.decode_with_output_nodes(tpg_program.genome, ml, ma, si).programs[1]
    names = String[string(op.fn.name) for op in program]
    return join(names, " → ")
end

_dot_escape(s) = replace(string(s), "&" => "&amp;", "<" => "&lt;", ">" => "&gt;", "\"" => "&quot;")
_team_id(t) = t isa TeamID ? t : TeamID(t)

"""
Teams to draw (all, or those reachable from `root`) and teams to highlight
(those reachable from `highlight`, or none).
"""
function _plot_scope(tpg::TangledProgramGraph, root, highlight)
    reachable(t) = first(_traverse_tpg_from_roots(tpg, _team_id(t)))
    team_ids = isnothing(root) ? collect(keys(tpg.teams)) : collect(reachable(root))
    sort!(team_ids, by = t -> t.val)
    highlighted = isnothing(highlight) ? Set{TeamID}() : Set{TeamID}(reachable(highlight))
    return team_ids, highlighted
end

"""
    tpg_to_dot(tpg; root = nothing, highlight = nothing, describe = p -> "")::String

Graphviz DOT source for `tpg`, teams drawn as tables left to right. With
`root`, only the teams reachable from it are drawn. With `highlight`, the
teams and edges reachable from that root are colored and the rest is faded.
Each team lists its programs: id, where the program leads (a class/action when
it is a leaf, or the next team) and `describe(p)`.
"""
function tpg_to_dot(tpg::TangledProgramGraph; root = nothing, highlight = nothing, describe = p -> "")::String
    team_ids, highlighted = _plot_scope(tpg, root, highlight)
    fading = !isnothing(highlight)

    io = IOBuffer()
    println(io, "digraph TPG {")
    println(io, "  rankdir=LR;")
    println(io, "  node [shape=plaintext, fontname=\"Helvetica\", fontsize=10];")
    println(io, "  edge [arrowhead=vee];")
    for team_id in team_ids
        team = tpg.teams[team_id]
        is_root = team_id in tpg.root_teams
        on_path = team_id in highlighted
        faded = fading && !on_path
        header_color = if fading && team_id == _team_id(highlight)
            "#F4B183"
        elseif faded
            "#F2F2F2"
        else
            is_root ? "#9CC3E6" : "#D9D9D9"
        end
        title = fading && team_id == _team_id(highlight) ? " (best root)" : is_root ? " (root)" : ""
        font = faded ? "#BBBBBB" : "#000000"
        println(io, "  T$(team_id.val) [fontcolor=\"$font\", label=<")
        println(io, "    <TABLE BORDER=\"1\" CELLBORDER=\"1\" CELLSPACING=\"0\" CELLPADDING=\"4\" COLOR=\"$(faded ? "#DDDDDD" : "#000000")\">")
        println(io, "    <TR><TD COLSPAN=\"3\" BGCOLOR=\"$header_color\"><B>Team $(team_id.val)$title</B></TD></TR>")
        for p in team.programs
            next_team = get(team.action_map, p.id, nothing)
            target = isnothing(next_team) ? "class $(p.action)" : "→ T$(next_team.val)"
            row_color = faded ? "#FFFFFF" : isnothing(next_team) ? "#E2F0D9" : "#FFFFFF"
            println(
                io, "    <TR><TD PORT=\"p$(p.id.val)\" BGCOLOR=\"$row_color\">P$(p.id.val)</TD>",
                "<TD BGCOLOR=\"$row_color\">$(_dot_escape(target))</TD>",
                "<TD ALIGN=\"LEFT\" BGCOLOR=\"$row_color\">$(_dot_escape(describe(p)))</TD></TR>"
            )
        end
        println(io, "    </TABLE>>];")
    end
    team_set = Set(team_ids)
    for team_id in team_ids
        for (program_id, next_team_id) in tpg.teams[team_id].action_map
            next_team_id in team_set || continue
            color = fading && !(team_id in highlighted) ? "#DDDDDD" : fading ? "#C00000" : "#555555"
            println(io, "  T$(team_id.val):p$(program_id.val) -> T$(next_team_id.val) [color=\"$color\"];")
        end
    end
    println(io, "}")
    return String(take!(io))
end

"""
    tpg_to_network_dot(tpg; root = nothing, highlight = nothing, describe = p -> "")::String

Classic TPG drawing: teams are circles, elite roots black diamonds, every program of a team
is a small node on the edge leaving that team, leading either to the next team
or to a dark box with its class. A side table maps program ids to
`describe(p)` (programs on the highlighted path only, when highlighting).
With `highlight`, that root is red and everything it can reach is drawn in
full color; the rest of the graph is faded. Meant for a radial layout (`twopi`).
"""
function tpg_to_network_dot(tpg::TangledProgramGraph; root = nothing, highlight = nothing, describe = p -> "")::String
    team_ids, highlighted = _plot_scope(tpg, root, highlight)
    team_set = Set(team_ids)
    fading = !isnothing(highlight)
    center = !isnothing(highlight) ? highlight : root

    io = IOBuffer()
    println(io, "digraph TPG {")
    println(io, "  overlap=false; splines=true; outputorder=edgesfirst;")
    isnothing(center) || println(io, "  root=T$(_team_id(center).val);")
    println(io, "  node [fontname=\"Helvetica\", fontsize=9];")
    println(io, "  edge [arrowsize=0.5];")
    table_programs = Dict{ProgramID, TPGProgram}()
    for team_id in team_ids
        is_root = team_id in tpg.root_teams
        faded = fading && !(team_id in highlighted)
        fill = if fading && team_id == _team_id(highlight)
            "#D62728"
        elseif faded
            is_root ? "#9A9A9A" : "#E6E6E6"
        elseif fading
            is_root ? "black" : "#F08080" # on the highlighted path
        else
            is_root ? "black" : "#A0A0A0"
        end
        font = faded ? "#CCCCCC" : "#000000"
        edge_color = faded ? "#E6E6E6" : fading ? "#C00000" : "#AAAAAA"
        line = faded ? "#DDDDDD" : "#666666"
        shape, size = is_root ? ("diamond", 0.4) : ("circle", 0.25) # elite roots are diamonds
        println(io, "  T$(team_id.val) [shape=$shape, style=filled, fillcolor=\"$fill\", color=\"$fill\", label=\"\", width=$size, height=$size, xlabel=\"T$(team_id.val)\", fontcolor=\"$font\"];")
        for p in tpg.teams[team_id].programs
            faded || (table_programs[p.id] = p)
            node = "T$(team_id.val)_P$(p.id.val)"
            println(io, "  $node [shape=box, style=\"rounded\", color=\"$line\", fontcolor=\"$font\", label=\"P$(p.id.val)\", width=0.2, height=0.15, margin=0.03];")
            println(io, "  T$(team_id.val) -> $node [color=\"$edge_color\"];")
            next_team = get(tpg.teams[team_id].action_map, p.id, nothing)
            if !isnothing(next_team) && next_team in team_set
                println(io, "  $node -> T$(next_team.val) [color=\"$edge_color\"];")
            else
                leaf = "$(node)_A"
                leaf_fill = faded ? "#DDDDDD" : "#555555"
                println(io, "  $leaf [shape=box, style=filled, fillcolor=\"$leaf_fill\", color=\"$leaf_fill\", fontcolor=white, label=\"c$(_dot_escape(p.action))\", width=0.2, height=0.15, margin=0.03];")
                println(io, "  $node -> $leaf [color=\"$edge_color\"];")
            end
        end
    end
    # Side table: program id -> what it computes
    println(io, "  programs [shape=plaintext, label=<")
    println(io, "    <TABLE BORDER=\"1\" CELLBORDER=\"0\" CELLSPACING=\"0\" CELLPADDING=\"3\">")
    println(io, "    <TR><TD BGCOLOR=\"#D9D9D9\"><B>Program</B></TD><TD BGCOLOR=\"#D9D9D9\" ALIGN=\"LEFT\"><B>Functions</B></TD></TR>")
    for pid in sort!(collect(keys(table_programs)), by = p -> p.val)
        println(io, "    <TR><TD>P$(pid.val)</TD><TD ALIGN=\"LEFT\">$(_dot_escape(describe(table_programs[pid])))</TD></TR>")
    end
    println(io, "    </TABLE>>];")
    println(io, "}")
    return String(take!(io))
end

"""
    plot_tpg(tpg, filename; root = nothing, highlight = nothing, describe = p -> "", style = :table)

Write the DOT source to `filename`. A `.dot` filename keeps the source only;
`.svg`, `.png` or `.pdf` also renders it with Graphviz, keeping the `.dot`
source next to it. `root` limits the drawing to what that root reaches;
`highlight` colors what that root reaches and fades the rest. `style = :table`
draws teams as tables (left to right, `dot`); `style = :network` draws the
classic radial TPG (`twopi`).
"""
function plot_tpg(
        tpg::TangledProgramGraph, filename::AbstractString;
        root = nothing, highlight = nothing, describe = p -> "", style::Symbol = :table
    )
    dot_source, engine = if style === :table
        tpg_to_dot(tpg; root = root, highlight = highlight, describe = describe), "dot"
    elseif style === :network
        tpg_to_network_dot(tpg; root = root, highlight = highlight, describe = describe), "twopi"
    else
        error("Unknown TPG plot style $style. Use :table or :network.")
    end
    base, ext = splitext(filename)
    dot_file = base * ".dot"
    write(dot_file, dot_source)
    if ext != ".dot"
        run(`$engine -T$(ext[2:end]) $dot_file -o $filename`)
    end
    @info "TPG graph saved to $filename"
    return filename
end
