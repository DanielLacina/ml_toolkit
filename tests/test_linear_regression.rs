use ml_toolkit::algorithms::linear_regression::linear_regression::LinearRegression;
use ml_toolkit::dataframe::csv::df_from_csv;
use ml_toolkit::dataframe::{DataFrame, DataType};
use ml_toolkit::inference::inference::rmse;
// use ml_toolkit::pipeline::one_hot_encoder::df_one_hot_encoded;
use ml_toolkit::pipeline::encoders::one_hot_encoder::OneHotEncoder;
use ml_toolkit::pipeline::imputers::imputer::{Imputer, ImputerStrategy};
use ml_toolkit::pipeline::pipeline::*;
use ml_toolkit::pipeline::scalars::standard_scalar::StandardScalar;
use ml_toolkit::pipeline::transformers::Transformer;

#[test]
fn test_linear_regression() {
    let filename = "housing.csv";
    let mut df = df_from_csv(filename, None);
    let label = "median_house_value";
    let labels = df
        .get_columns_as_df(&vec![label.to_string()])
        .as_matrix(false);
    df.remove_column(label);
    let one_hot_encoder: Box<dyn Transformer> = Box::new(OneHotEncoder::new(true));
    let std_scalar: Box<dyn Transformer> = Box::new(StandardScalar::new());
    let imputer: Box<dyn Transformer> = Box::new(Imputer::new(&ImputerStrategy::Median));
    let cat_transformers = vec![one_hot_encoder];
    let cat_pipeline = CategoricalPipeline::new(cat_transformers);
    let num_transformers = vec![imputer, std_scalar];
    let num_pipeline = NumericalPipeline::new(num_transformers);
    let column_transformer = ColumnTransformer::new(num_pipeline, cat_pipeline);
    let inputs = column_transformer.transform(&df);
    let mut linear_regression = LinearRegression::new(0.0);
    linear_regression.fit(&inputs, &labels);
    let weights = linear_regression.weights();
    assert!(weights.len() == inputs[0].len());
    let bias = linear_regression.bias();
    assert!(bias != 0.0);
    let prediction = linear_regression.predict(&inputs);
}
