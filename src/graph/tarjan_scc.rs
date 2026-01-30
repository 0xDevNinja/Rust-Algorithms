//! Tarjan's strongly-connected components algorithm. Single-pass DFS with
//! discovery / low-link bookkeeping. O(V + E).

/// Returns the strongly-connected components of a directed graph as a list
/// of node lists. Each component is sorted ascending; the components are
/// returned in reverse-topological order (sinks of the condensation first).
pub fn tarjan_scc(graph: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let n = graph.len();
    let mut state = TarjanState {
        disc: vec![usize::MAX; n],
        low: vec![0; n],
        on_stack: vec![false; n],
        stack: Vec::new(),
        components: Vec::new(),
        index: 0,
    };
    for u in 0..n {
        if state.disc[u] == usize::MAX {
            strongconnect(graph, u, &mut state);
        }
    }
    state.components
}

struct TarjanState {
    disc: Vec<usize>,
    low: Vec<usize>,
    on_stack: Vec<bool>,
    stack: Vec<usize>,
    components: Vec<Vec<usize>>,
    index: usize,
}

fn strongconnect(graph: &[Vec<usize>], u: usize, st: &mut TarjanState) {
    st.disc[u] = st.index;
    st.low[u] = st.index;
    st.index += 1;
    st.stack.push(u);
    st.on_stack[u] = true;

    for &v in &graph[u] {
        if st.disc[v] == usize::MAX {
            strongconnect(graph, v, st);
            st.low[u] = st.low[u].min(st.low[v]);
        } else if st.on_stack[v] {
            st.low[u] = st.low[u].min(st.disc[v]);
        }
    }

    if st.low[u] == st.disc[u] {
        let mut component = Vec::new();
        loop {
            let v = st.stack.pop().unwrap();
            st.on_stack[v] = false;
            component.push(v);
            if v == u {
                break;
            }
        }
        component.sort_unstable();
        st.components.push(component);
    }
}

#[cfg(test)]
mod tests {
    use super::tarjan_scc;

    fn sort_components(mut comps: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
        comps.iter_mut().for_each(|c| c.sort_unstable());
        comps.sort_by_key(|c| c[0]);
        comps
    }

    #[test]
    fn empty() {
        let g: Vec<Vec<usize>> = vec![];
        assert!(tarjan_scc(&g).is_empty());
    }

    #[test]
    fn single_node() {
        let g = vec![vec![]];
        assert_eq!(tarjan_scc(&g), vec![vec![0]]);
    }

    #[test]
    fn two_disconnected() {
        let g = vec![vec![], vec![]];
        let c = sort_components(tarjan_scc(&g));
        assert_eq!(c, vec![vec![0], vec![1]]);
    }

    #[test]
    fn classic_clrs_example() {
        // Edges: a->b, b->c, b->e, b->f, c->d, c->g, d->c, d->h,
        // e->a, e->f, f->g, g->f, g->h, h->h
        // Vertices a..h = 0..7
        let g = vec![
            vec![1],       // a
            vec![2, 4, 5], // b
            vec![3, 6],    // c
            vec![2, 7],    // d
            vec![0, 5],    // e
            vec![6],       // f
            vec![5, 7],    // g
            vec![7],       // h
        ];
        let c = sort_components(tarjan_scc(&g));
        assert_eq!(c, vec![vec![0, 1, 4], vec![2, 3], vec![5, 6], vec![7]]);
    }

    #[test]
    fn self_loop() {
        let g = vec![vec![0]];
        assert_eq!(tarjan_scc(&g), vec![vec![0]]);
    }

    #[test]
    fn one_big_cycle() {
        let g = vec![vec![1], vec![2], vec![3], vec![0]];
        let c = sort_components(tarjan_scc(&g));
        assert_eq!(c, vec![vec![0, 1, 2, 3]]);
    }
}
