# --- TPG Plotting ---
"""
    plot_tpg(tpg::TangledProgramGraph; filename::String = "tpg_graph.svg", layout_engine::String = "dot")
Generates a Graphviz visualization of the TangledProgramGraph and saves it to a file.
Teams are represented as nodes. Programs within teams are listed as labels, including their assigned action.
Connections between teams are shown as edges. Root teams are highlighted.
"""
function plot_tpg(tpg::TangledProgramGraph; filename::String = "tpg_graph.svg", layout_engine::String = "dot")
    dot_str_buffer = IOBuffer()
    println(dot_str_buffer, "digraph TPG {")
    println(dot_str_buffer, "  rankdir=LR;") # Left to Right layout

    # Define node styles for teams
    println(dot_str_buffer, "  node [shape=box, style=\"rounded,filled\", fillcolor=\"#E6E6FA\", fontname=\"Helvetica\"];")

    # Nodes for teams
    for (team_id, team) in tpg.teams
        programs_str = ""
        for p in team.programs
            action_str = p.action !== nothing ? " (Action: $(p.action))" : ""
            programs_str *= "P$(p.id.val)$(action_str)\\n"
        end
        label = "Team $(team_id.val)\\n$(programs_str)"

        # Highlight root teams
        if team_id in tpg.root_teams
            println(dot_str_buffer, "  Team_$(team_id.val) [label=\"$(label)\", fillcolor=\"#B0E0E6\", penwidth=2, color=\"#4682B4\"];")
        else
            println(dot_str_buffer, "  Team_$(team_id.val) [label=\"$(label)\"];")
        end
    end

    # Edges for connections between teams based on action_map
    println(dot_str_buffer, "  edge [color=\"#696969\", arrowhead=vee];")
    for (team_id, team) in tpg.teams
        for (program_id, next_team_id) in team.action_map
            if haskey(tpg.teams, next_team_id)
                # Label the edge with the program that triggered the transition
                println(dot_str_buffer, "  Team_$(team_id.val) -> Team_$(next_team_id.val) [label=\"P$(program_id.val)\"];")
            end
        end
    end

    println(dot_str_buffer, "}")
    dot_string = String(take!(dot_str_buffer))

    # Save to file
    Graphviz.render(Graphviz.Graph(dot_string), filename, layout = layout_engine)
    return @info "TPG graph saved to $(filename)"
end
