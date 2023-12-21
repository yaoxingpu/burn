#[burn_tensor_testgen::testgen(div)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Int, Tensor};

    #[test]
    fn should_support_div_ops() {
        let tensor_1 = Tensor::<TestBackend, 2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor_2 = Tensor::<TestBackend, 2>::from([[1.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let output = tensor_1 / tensor_2;

        let data_actual = output.into_data();
        let data_expected = Data::from([[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]]);
        data_expected.assert_approx_eq(&data_actual, 3);
    }

    #[test]
    fn test_div_broadcast() {
        let tensor_1 = Tensor::<TestBackend, 2>::from([[0.0, 1.0, 2.0]]);
        let tensor_2 = Tensor::<TestBackend, 2>::from([[1.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let data_actual = (tensor_1 / tensor_2).into_data();

        let data_expected = Data::from([[0.0, 1.0, 1.0], [0.0, 0.25, 0.4]]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_div_scalar_ops() {
        let scalar = 2.0;
        let tensor = Tensor::<TestBackend, 2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let output = tensor / scalar;

        let data_actual = output.into_data();
        let data_expected = Data::from([[0.0, 0.5, 1.0], [1.5, 2.0, 2.5]]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_div_ops_int() {
        let tensor_1 = Tensor::<TestBackend, 2, Int>::from([[0, 1, 2], [3, 4, 5]]);
        let tensor_2 = Tensor::<TestBackend, 2, Int>::from([[1, 1, 2], [1, 1, 2]]);

        let output = tensor_1 / tensor_2;

        let data_actual = output.into_data();
        let data_expected = Data::from([[0, 1, 1], [3, 4, 2]]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn test_div_broadcast_int() {
        let tensor_1 = Tensor::<TestBackend, 2, Int>::from([[0, 1, 2]]);
        let tensor_2 = Tensor::<TestBackend, 2, Int>::from([[1, 1, 2], [3, 4, 5]]);

        let data_actual = (tensor_1 / tensor_2).into_data();

        let data_expected = Data::from([[0, 1, 1], [0, 0, 0]]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_div_scalar_ops_int() {
        let scalar = 2;
        let tensor = Tensor::<TestBackend, 2, Int>::from([[0, 1, 2], [3, 4, 5]]);

        let output = tensor / scalar;

        let data_actual = output.into_data();
        let data_expected = Data::from([[0, 0, 1], [1, 2, 2]]);
        assert_eq!(data_expected, data_actual);
    }
}
