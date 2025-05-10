use ml_toolkit::algorithms::linear_regression::linear_regression::LinearRegression;
use ml_toolkit::dataframe::csv::df_from_csv;
use ml_toolkit::inference::inference::rmse;
use ml_toolkit::pipeline::encoders::one_hot_encoder::OneHotEncoder;
use ml_toolkit::pipeline::imputers::imputer::{Imputer, ImputerStrategy};
use ml_toolkit::pipeline::pipeline::*;
use ml_toolkit::pipeline::scalars::standard_scalar::StandardScalar;
use ml_toolkit::pipeline::transformers::Transformer;
use ml_toolkit::sampling::sampling::StratifiedShuffleSplit;

#[test]
fn test_linear_regression() {
    let filename = "housing.csv";
    let df = df_from_csv(filename, None);
    let stratified_shuffle_split =
        StratifiedShuffleSplit::new(0.2, &vec![("median_income".to_string(), 6)]);
    let (train_indices, test_indices) = stratified_shuffle_split.split(&df);
    let (train_set, test_set) = (
        df.get_rows_as_df(&train_indices),
        df.get_rows_as_df(&test_indices),
    );
    let (mut train_features, mut test_features) = (train_set.clone(), test_set.clone());
    let label = "median_house_value";
    train_features.remove_column(label);
    test_features.remove_column(label);
    let (train_labels, test_labels) = (
        train_set
            .get_columns_as_df(&vec![label.to_string()])
            .as_matrix(false),
        test_set
            .get_columns_as_df(&vec![label.to_string()])
            .as_matrix(false),
    );
    let one_hot_encoder: Box<dyn Transformer> = Box::new(OneHotEncoder::new(true));
    let std_scalar: Box<dyn Transformer> = Box::new(StandardScalar::new());
    let imputer: Box<dyn Transformer> = Box::new(Imputer::new(&ImputerStrategy::Median));
    let cat_transformers = vec![one_hot_encoder];
    let cat_pipeline = CategoricalPipeline::new(cat_transformers);
    let num_transformers = vec![imputer, std_scalar];
    let num_pipeline = NumericalPipeline::new(num_transformers);
    let column_transformer = ColumnTransformer::new(num_pipeline, cat_pipeline);
    let (train_inputs, test_inputs) = (
        column_transformer.transform(&train_features),
        column_transformer.transform(&test_features),
    );
    let mut linear_regression = LinearRegression::new(0.0);
    linear_regression.fit(&train_inputs, &train_labels);
    assert!(linear_regression.weights().len() == train_inputs.get(0).len());
    assert!(linear_regression.bias() != 0.0);
    let (train_predictions, test_predictions) = (
        linear_regression.predict(&train_inputs),
        linear_regression.predict(&test_inputs),
    );
    let (train_labels, test_labels) = (
        train_labels.matrix()
            .into_iter()
            .map(|label| label.get(0))
            .collect::<Vec<f32>>(),
        test_labels.matrix()
            .into_iter()
            .map(|label| label.get(0))
            .collect::<Vec<f32>>(),
    );
    assert!(train_predictions.len() == train_labels.len());
    assert!(test_predictions.len() == test_labels.len());
    let (train_rmse, test_rmse) = (
        rmse(&train_predictions.vector(), &train_labels),
        rmse(&test_predictions.vector(), &test_labels),
    );
    assert!(train_rmse < 70000.0);
    assert!(test_rmse < 70000.0);
}
