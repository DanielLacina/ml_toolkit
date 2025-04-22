mod algorithms;
mod dataframe;
mod inference;
mod linear_algebra;
mod pipeline;
mod sampling;
use algorithms::linear_regression::linear_regression::LinearRegression;
use dataframe::csv::df_from_csv;
use inference::inference::rmse;
// use pipeline::one_hot_encoder::df_one_hot_encoded;
use pipeline::encoders::one_hot_encoder::OneHotEncoder;
use pipeline::imputers::imputer::{Imputer, ImputerStrategy};
use pipeline::pipeline::*;
use pipeline::scalars::standard_scalar::StandardScalar;
use sampling::sampling::StratifiedShuffleSplit;

fn main() {
    let filename = "housing.csv";
    let df = df_from_csv(filename, None);
    let stratified_shuffle_split =
        StratifiedShuffleSplit::new(0.2, &vec![("median_income".to_string(), 100)]);
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
    let one_hot_encoder = OneHotEncoder::new();
    let std_scalar = StandardScalar::new();
    let imputer = Imputer::new(&ImputerStrategy::Median);
    let cat_pipeline = CategoricalPipeline::new(Box::new(one_hot_encoder));
    let num_pipeline = NumericalPipeline::new(imputer, Some(Box::new(std_scalar)));
    let column_transformer = ColumnTransformer::new(num_pipeline, cat_pipeline);
    let (train_inputs, test_inputs) = (
        column_transformer.transform(&train_features),
        column_transformer.transform(&test_features),
    );
    let mut linear_regression = LinearRegression::new(25.0);
    linear_regression.fit(&train_inputs, &train_labels);
    let (train_predictions, test_predictions) = (
        linear_regression.predict(&train_inputs),
        linear_regression.predict(&test_inputs),
    );
    let (train_labels, test_labels) = (
        train_labels
            .into_iter()
            .map(|label| label[0])
            .collect::<Vec<f32>>(),
        test_labels
            .into_iter()
            .map(|label| label[0])
            .collect::<Vec<f32>>(),
    );
    let (train_rmse, test_rmse) = (
        rmse(&train_predictions, &train_labels),
        rmse(&test_predictions, &test_labels),
    );
    println!("{}, {}", train_rmse, test_rmse);
}
