// use std::{collections::HashMap, fmt::Debug, sync::Arc};

// use super::{Node, NodeID, NodeRef};

// pub enum OperationBottleneck {
//     ComputeBound,
//     MemoryBound,
// }

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

// pub trait State: Sync + Send + Debug {}

// #[derive(Default, Debug)]
// pub struct NodeStates {
//     hashmap: HashMap<NodeID, StateBoxed>,
// }

// pub type StateBoxed = Box<dyn State>;

// impl NodeStates {
//     pub fn get_input(&self, node: NodeRef) -> StateBoxed {
//         node.parents
//             .iter()
//             .map(|parent| self.get_output(parent))
//             .collect()
//     }

//     pub fn get_output(&self, node: Node) -> StateBoxed {
//         match self.hashmap.get(node.id) {
//             Some(state) => state,
//             None => {
//                 self.get_input(node);
//                 //node forward does not exist
//                 Box::new(node.forward(inputs))
//             }
//         }
//     }
// }
