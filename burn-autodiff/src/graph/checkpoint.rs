use std::{collections::HashMap, fmt::Debug};

use burn_tensor::backend::Backend;

use crate::ops::Ops;

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

pub trait State: Sync + Send + Debug {}

#[derive(new, Debug)]
pub struct StateStruct<B: Backend, const D: usize> {
    tensor: B::TensorPrimitive<D>,
}

impl<B: Backend, const D: usize> State for StateStruct<B, D> {}

#[derive(Default, Debug)]
pub struct NodeStates {
    hashmap: HashMap<NodeID, StateBoxed>,
}

pub type StateBoxed = Box<dyn State>;

impl NodeStates {
    pub fn register(mut self, node_id: NodeID, state: StateBoxed) {
        self.hashmap.insert(node_id, state);
    }

    pub fn get_input(&self, node: NodeRef) -> Vec<StateBoxed> {
        node.parents
            .iter()
            .map(|parent| self.get_output(parent))
            .collect()
    }

    pub fn get_output(&self, node_id: &NodeID) -> StateBoxed {
        match self.hashmap.remove(node_id) {
            Some(state) => state,
            None => {
                // TODO not <_, _, 1>
                let ops: Ops<_, _, 1> = self.get_ops_from_node_id(node_id);
                let inputs = self.get_input(ops.node);
                //node forward does not exist
                Box::new(ops.forward(inputs))
            }
        }
    }

    // NodeStates must have access to a mapping from NodeRef/NodeID to Ops
    // Otherwise how to get parents just with ids?
    // And how to do the forward pass ?
    fn get_ops_from_node_id<I, O, const N: usize>(&self, node_id: &NodeID) -> Ops<I, O, N> {
        todo!()
    }
}
