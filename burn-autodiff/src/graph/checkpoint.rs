use std::{collections::HashMap, fmt::Debug};

use burn_tensor::backend::Backend;
use downcast_rs::Downcast;

use crate::ops::{Operation, Ops, OpsSpec};

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

pub trait State: Send + Sync + Debug + 'static + Downcast {}

#[derive(new, Debug, Clone)]
pub struct StateStruct<B: Backend, const D: usize, const N: usize> {
    pub tensors: Vec<B::TensorPrimitive<D>>, // size N
                                             // pub tensors: [B::TensorPrimitive<D>; N],
}

impl<B: Backend, const D: usize, const N: usize> State for StateStruct<B, D, N> {}

// Not sure necessary, delete if possible
#[derive(new, Debug, Clone)]
pub struct StateNull {}
impl State for StateNull {}

pub type StateBoxed = Box<dyn State>;

pub type OperationBoxed = Box<dyn Operation>;

#[derive(Default, Debug)]
pub struct NodeStates {
    state_hashmap: HashMap<NodeID, StateBoxed>,
    operation_hashmap: HashMap<NodeID, OperationBoxed>,
}

impl NodeStates {
    pub fn register(mut self, node_id: NodeID, state: StateBoxed, operation: OperationBoxed) {
        self.state_hashmap.insert(node_id, state);
        self.operation_hashmap.insert(node_id, operation);
    }

    pub fn get_input<B, OS, I, O, const D: usize, const N: usize>(&self, node: NodeRef) -> I
    where
        B: Backend,
        OS: OpsSpec<B, D, N, Input = I, Output = O>,
        I: State,
        O: State,
    {
        let x: Vec<Box<dyn State>> = node
            .parents
            .iter()
            .map(|parent| self.get_output::<B, OS, I, O, D, N>(parent))
            .collect();

        *outputs_to_input::<B, D, N>(x)
            .as_any()
            .downcast_ref::<I>()
            .expect("Downcast failed")
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
        match self.state_hashmap.remove(node_id) {
            Some(state) => state,
            None => {
                // let ops: Ops<B, OS, I, O, D, N> = self.get_ops_from_node_id(node_id);
                let operation: &OperationBoxed =
                    self.get_ops_from_node_id::<B, OS, I, O, D, N>(node_id);
                let inputs = self.get_input::<B, OS, I, O, D, N>(operation.node());
                operation.forward(Box::new(inputs))
            }
        }
    }

    // maybe inline
    fn get_ops_from_node_id<B, OS, I, O, const D: usize, const N: usize>(
        &self,
        node_id: &NodeID,
    ) -> &OperationBoxed
    // ) -> Ops<B, OS, I, O, D, N>
    where
        OS: OpsSpec<B, D, N, Input = I, Output = O>,
        B: Backend,
        I: State,
        O: State,
    {
        self.operation_hashmap.get(node_id).unwrap()
    }
}

fn outputs_to_input<B: Backend, const D: usize, const N: usize>(
    outputs: Vec<StateBoxed>,
) -> StateBoxed {
    let x: Vec<StateStruct<B, D, N>> = outputs
        .iter()
        .map(|out| {
            *out.as_any()
                .downcast_ref::<StateStruct<B, D, N>>()
                .expect("Downcast failed")
        })
        .collect();
    let y: Vec<B::TensorPrimitive<D>> = x
        .iter()
        .map(|state_struct| state_struct.tensors[0])
        .collect();
    Box::new(StateStruct::<B, D, N>::new(y))
}
