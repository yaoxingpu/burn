@group(0)
@binding(0)
var<storage, read> input: array<u32>;

@group(0)
@binding(1)
var<storage, read_write> output: array<{{ output_elem }}>;

@compute
@workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>, 
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let id = global_id.y * (num_workgroups.x * {{ workgroup_size_x }}u) + global_id.x;
    if (input[id] > 0u){
        output[id] = {{ output_elem }}(1);
    } else {
        output[id] = {{ output_elem }}(0);
    }
}
