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
use ml_toolkit::sampling::sampling::StratifiedShuffleSplit;
use ml_toolkit::pipeline::polynomial_features::polynomial_features::PolynomialFeatures;

pub struct CombinedAttributesAdder;

impl CombinedAttributesAdder {
    pub fn new() -> Self {
        Self
    }
}

impl Transformer for CombinedAttributesAdder {
    fn transform(&self, df: &DataFrame, column_names: &Vec<String>) -> DataFrame {
        let mut df = df.clone();
        let total_rooms = "total_rooms".to_string();
        let households = "households".to_string();
        let population = "population".to_string();
        let total_bedrooms = "total_bedrooms".to_string();
        let rooms_per_households = df.divide_columns(&total_rooms, &households);
        let population_per_households = df.divide_columns(&population, &households);
        let bedrooms_per_room = df.divide_columns(&total_bedrooms, &total_rooms);
        df.insert_column(
            "rooms_per_household",
            &rooms_per_households,
            &DataType::Float,
        );
        df.insert_column(
            "population_per_household",
            &population_per_households,
            &DataType::Float,
        );
        df.insert_column("bedrooms_per_room", &bedrooms_per_room, &DataType::Float);
        return df;
    }
}
fn main() {
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
    let polynomial_features = PolynomialFeatures::new(2);
    let columns: Vec<String> = train_features.numeric_columns().into_iter().map(|column| column.clone()).collect(); 
    let imputer: Box<dyn Transformer> = Box::new(Imputer::new(&ImputerStrategy::Median));
    let train_features = imputer.transform(&train_features, &columns); 
    let test_features = imputer.transform(&test_features, &columns); 
    let train_features = polynomial_features.transform(&train_features, &columns);
    let test_features = polynomial_features.transform(&test_features, &columns);
    let (train_labels, test_labels) = (
        train_set
            .get_columns_as_df(&vec![label.to_string()])
            .as_matrix(false),
        test_set
            .get_columns_as_df(&vec![label.to_string()])
            .as_matrix(false),
    );
    let combined_attr_adder: Box<dyn Transformer> = Box::new(CombinedAttributesAdder::new());
    let one_hot_encoder: Box<dyn Transformer> = Box::new(OneHotEncoder::new(true));
    let std_scalar: Box<dyn Transformer> = Box::new(StandardScalar::new());
    let cat_transformers = vec![one_hot_encoder];
    let cat_pipeline = CategoricalPipeline::new(cat_transformers);
    let num_transformers = vec![combined_attr_adder, std_scalar];
    let num_pipeline = NumericalPipeline::new(num_transformers);
    let column_transformer = ColumnTransformer::new(num_pipeline, cat_pipeline);
    let (train_inputs, test_inputs) = (
        column_transformer.transform(&train_features),
        column_transformer.transform(&test_features),
    );
    let mut linear_regression = LinearRegression::new(0.0);
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
