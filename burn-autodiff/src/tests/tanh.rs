#[burn_tensor_testgen::testgen(ad_tanh)]
mod tests {
    use super::*;
    use burn_tensor::Data;

    #[test]
    fn should_diff_tanh() {
        let tensor_1 = TestAutodiffTensor::from([[0.0, 1.0], [3.0, 4.0]]).require_grad();
        let tensor_2 = TestAutodiffTensor::from([[6.0, 7.0], [9.0, 10.0]]).require_grad();

        let tensor_3 = tensor_1.clone().matmul(tensor_2.clone().tanh());
        let tensor_4 = tensor_3.matmul(tensor_2.clone());
        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        grad_1
            .to_data()
            .assert_approx_eq(&Data::from([[32.0, 32.0], [32.0, 32.0]]), 3);
        grad_2
            .to_data()
            .assert_approx_eq(&Data::from([[8.00092, 8.000153], [8.000003, 7.999995]]), 3);
    }
}
