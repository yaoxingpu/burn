use std::{collections::HashMap, fmt::Debug};

use burn_tensor::backend::Backend;

use crate::ops::{Ops, OpsSpec};

use super::{NodeID, NodeRef};

// use super::{Node, NodeID, NodeRef};

pub enum Bottleneck {
    ComputeBound,
    MemoryBound,
}

// // /// Some of those are just ideas. ComputeBoundOnly is the goal
// // pub enum CheckpointingStrategy {
// //     /// All states are saved (comparable to what we had before)
// //     All,
// //     /// No state is saved, backward is very compute heavy but memory used is limited
// //     None,
// //     /// Only operations whose workload is significant have their output saved
// //     ComputeBoundOnly,
// //     /// Every n operations (although does not make much sense in a tree)
// //     Periodic(n),
// //     /// Manually tag some operations to be checkpointed (pytorch style). Will need a new backend method to tell the next operation must be checkpointed
// //     Manual,
// // }

pub trait State: Send + Sync + Debug + 'static {}

#[derive(new, Debug, Clone)]
pub struct StateStruct<B: Backend, const D: usize, const N: usize> {
    pub tensors: [B::TensorPrimitive<D>; N],
}

impl<B: Backend, const D: usize, const N: usize> State for StateStruct<B, D, N> {}

// Not sure necessary, delete if possible
#[derive(new, Debug, Clone)]
pub struct StateNull {}
impl State for StateNull {}

#[derive(Default, Debug)]
pub struct NodeStates {
    hashmap: HashMap<NodeID, StateBoxed>,
}

pub type StateBoxed = Box<dyn State>;

impl NodeStates {
    pub fn register(mut self, node_id: NodeID, state: StateBoxed) {
        self.hashmap.insert(node_id, state);
    }

    pub fn get_input<B, OS, I, O, const D: usize, const N: usize>(&self, node: NodeRef) -> I
    where
        B: Backend,
        OS: OpsSpec<B, D, N, Input = I, Output = O>,
        I: State,
        O: State,
    {
        node.parents
            .iter()
            .map(|parent| self.get_output::<B, OS, I, O, D, N>(parent))
            .collect()
    }

    pub fn get_output<B, OS, I, O, const D: usize, const N: usize>(
        &self,
        node_id: &NodeID,
    ) -> StateBoxed
    where
        B: Backend,
        OS: OpsSpec<B, D, N, Input = I, Output = O>,
        I: State,
        O: State,
    {
        match self.hashmap.remove(node_id) {
            Some(state) => state,
            None => {
                let ops: Ops<B, OS, I, O, D, N> = self.get_ops_from_node_id(node_id);
                let inputs = self.get_input::<B, OS, I, O, D, N>(ops.node);
                Box::new(ops.forward(inputs))
            }
        }
    }

    // NodeStates must have access to a mapping from NodeRef/NodeID to Ops
    // Otherwise how to get parents just with ids?
    // And how to do the forward pass ?
    fn get_ops_from_node_id<B, OS, I, O, const D: usize, const N: usize>(
        &self,
        node_id: &NodeID,
    ) -> Ops<B, OS, I, O, D, N>
    where
        OS: OpsSpec<B, D, N, Input = I, Output = O>,
        B: Backend,
        I: State,
        O: State,
    {
        todo!()
    }
}

// STILL TO DO

// - Collect several Os into an I
// - node_id -> map of node_id -> Ops
//      when registering, pass a pointer to the ops too
