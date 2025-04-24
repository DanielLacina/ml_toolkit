use crate::dataframe::{DataFrame, DataTypeValue};
use std::collections::HashMap;
use std::collections::HashSet;

pub struct StratifiedShuffleSplit {
    test_size: f32,
    stratified_by: Vec<(String, usize)>,
}

impl StratifiedShuffleSplit {
    pub fn new(test_size: f32, stratified_by: &Vec<(String, usize)>) -> Self {
        if test_size < 0.0 || test_size > 1.0 {
            panic!("test size must be a percentage");
        }
        if stratified_by.len() == 0 {
            panic!("must provide at least one column to stratify the samples by");
        }
        if stratified_by.iter().any(|(_, num_bins)| *num_bins == 0) {
            panic!("cant divide data into 0 bins");
        }
        Self {
            test_size,
            stratified_by: stratified_by.clone(),
        }
    }

    fn bin_permutations_with_bin_nums_sorted(&self) -> Vec<Vec<u32>> {
        let mut bin_permutations = Vec::new();
        for (_, num_bins) in self.stratified_by.iter() {
            if bin_permutations.len() == 0 {
                for i in (0..*num_bins) {
                    bin_permutations.push(vec![i as u32]);
                }
            } else {
                let bin_permutations_copy = bin_permutations.clone();
                bin_permutations = Vec::new();
                for i in (0..*num_bins) {
                    let start = if i >= bin_permutations_copy.len() {
                        0
                    } else {
                        i
                    };
                    for bin in bin_permutations_copy[start..].iter() {
                        let mut bin = bin.clone();
                        bin.push(i as u32);
                        bin_permutations.push(bin);
                    }
                }
            }
        }
        for permutation in bin_permutations.iter_mut() {
            permutation.sort();
        }
        return bin_permutations;
    }

    fn sorted_bin_nums_from_df(&self, df: &DataFrame) -> Vec<(usize, Vec<u32>)> {
        let mut id_bin_permutations = Vec::new();
        let all_bins: Vec<Vec<(DataTypeValue, DataTypeValue, u32)>> = self
            .stratified_by
            .iter()
            .map(|(column_name, num_bins)| df.bins(&column_name, *num_bins))
            .collect();
        // ids should be ordered thus we iterate using range operator
        for i in (0..df.len()) {
            let mut bin_nums = Vec::new();
            for bins in all_bins.iter() {
                let bin_num = bins[i].2;
                bin_nums.push(bin_num);
            }
            bin_nums.sort();
            id_bin_permutations.push((i, bin_nums));
        }
        return id_bin_permutations;
    }

    pub fn split(&self, df: &DataFrame) -> (Vec<usize>, Vec<usize>) {
        let mut test_indices: Vec<usize> = Vec::new();
        let mut train_indices: Vec<usize> = Vec::new();
        let bin_permutations = self.bin_permutations_with_bin_nums_sorted();
        let mut bin_permutations_hashmap = HashMap::new();
        for bin_nums in bin_permutations.iter() {
            bin_permutations_hashmap.insert(bin_nums.clone(), Vec::new());
        }
        let id_bin_permutations = self.sorted_bin_nums_from_df(df);
        for (id, bin_nums) in id_bin_permutations.into_iter() {
            let bin_ids = bin_permutations_hashmap.get_mut(&bin_nums).unwrap();
            bin_ids.push(id);
        }
        for bin_nums in bin_permutations.iter() {
            let ids = bin_permutations_hashmap.get(bin_nums).unwrap();
            let divisor = (1.0 / self.test_size) as usize;
            for (i, id) in ids.iter().enumerate() {
                if i % divisor == 0 {
                    test_indices.push(*id)
                } else {
                    train_indices.push(*id);
                }
            }
        }
        return (train_indices, test_indices);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataframe::csv::df_from_csv;
    use std::collections::HashSet;

    fn test_stratified_shuffle_split(stratified_by: &Vec<(String, usize)>) {
        let filename = "housing.csv";
        let row_limit = 100;
        let df = df_from_csv(filename, Some(row_limit));
        let test_size = 0.2;
        let stratified_shuffle_split = StratifiedShuffleSplit::new(test_size, &stratified_by);
        let (train_indices, test_indices) = stratified_shuffle_split.split(&df);
        let mut train_indices_sorted = {
            let mut train_indices = train_indices.clone();
            train_indices.sort();
            train_indices
        };
        let mut test_indices_sorted = {
            let mut test_indices = test_indices.clone();
            test_indices.sort();
            test_indices
        };
        assert!(train_indices.len() == (df.len() as f32 * (1.0 - test_size)) as usize);
        assert!(test_indices.len() == (df.len() as f32 * test_size) as usize);
        assert!(train_indices_sorted != train_indices);
        assert!(test_indices_sorted != test_indices);
        let mut id_hashset = HashSet::new();
        let mut all_indices = Vec::new();
        all_indices.extend(train_indices);
        all_indices.extend(test_indices);
        assert!(all_indices.iter().all(|index| {
            if id_hashset.contains(index) {
                false
            } else {
                id_hashset.insert(index.clone());
                true
            }
        }));
    }
    #[test]
    fn test_simple_stratified_shuffle_split() {
        let stratified_by = vec![("median_income".to_string(), 5)];
        test_stratified_shuffle_split(&stratified_by);
    }

    #[test]
    fn test_multiple_stratified_shuffle_split() {
        let stratified_by = vec![
            ("median_income".to_string(), 5),
            ("households".to_string(), 2),
        ];
        test_stratified_shuffle_split(&stratified_by);
    }
}
