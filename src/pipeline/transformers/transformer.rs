use crate::dataframe::DataFrame;



pub trait Transformer {
    fn transform(&self, df: &DataFrame, column_names: &Vec<String>) -> DataFrame;
}