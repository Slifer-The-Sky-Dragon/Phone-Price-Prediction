#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>

using namespace std;

#define TRAIN_FILE_NAME "train.csv"
#define WEIGHTS_FILE_NAME "weights.csv"
#define DIVIDER '/'
#define DATASET_IND 1
#define SPACE ' '
#define COMMA ','
#define PERCANTAGE '%'

typedef vector < double > Feature_value_list;
typedef vector < double > Weight_list;
typedef int Mobile_type;
typedef double Price_class_bios;

struct Mobile_data{
	Feature_value_list feature;
	Mobile_type type;

	void add_feature(double feature_value){
		feature.push_back(feature_value);
	}
	int features_cnt(){
		return feature.size();
	}
};

struct Price_class{
	Weight_list weight;
	Price_class_bios bios;

	void add_feature_weight(double feature_weight){
		weight.push_back(feature_weight);
	}

	double calculate_class_score(Mobile_data cur_mobile_data){
		double result = 0;
		for(int feature_ind = 0 ; feature_ind < cur_mobile_data.features_cnt() ; feature_ind++){
			result += weight[feature_ind] * cur_mobile_data.feature[feature_ind];
		}
		return result + bios;
	}
};

typedef vector < Mobile_data > Mobiles_data_list;
typedef vector < Price_class > Price_class_list;

string replace_comma_with_space(string x){
	for(int i = 0 ; i < x.size() ; i++){
		if(x[i] == COMMA){
			x[i] = SPACE;
		}
	}
	return x;
}

string get_train_file_path(string dataset_path){
	if(dataset_path[dataset_path.size() - 1] == DIVIDER)
		return dataset_path + TRAIN_FILE_NAME;
	return dataset_path + DIVIDER + TRAIN_FILE_NAME;
}

string get_weight_file_path(string dataset_path){
	if(dataset_path[dataset_path.size() - 1] == DIVIDER)
		return dataset_path + WEIGHTS_FILE_NAME;
	return dataset_path + DIVIDER + WEIGHTS_FILE_NAME;
}

int find_number_of_features(ifstream& train_file){
	int result = 0;
	string name_line;
	getline(train_file , name_line);
	stringstream ss(replace_comma_with_space(name_line));

	string cur_feature_name;
	while(ss >> cur_feature_name)
		result++;
	return result - 1;
}

void read_and_store_data_from_train_file(ifstream& train_file , Mobiles_data_list& mobiles_data){
	int features_cnt = find_number_of_features(train_file);
	string data_line;
	while(getline(train_file , data_line)){
		stringstream ss(replace_comma_with_space(data_line));
		Mobile_data new_mobile_data;

		double tmp_feature_value;
		for(int feature_ind = 0 ; feature_ind < features_cnt ; feature_ind++){
			ss >> tmp_feature_value;
			new_mobile_data.add_feature(tmp_feature_value);
		}
		ss >> new_mobile_data.type;

		mobiles_data.push_back(new_mobile_data);
	}
}

void find_min_and_max_value_of_each_feature(Mobiles_data_list mobiles_data , Feature_value_list& min_val , 
													Feature_value_list& max_val){
	for(int mobile_ind = 0 ; mobile_ind < mobiles_data.size() ; mobile_ind++){
		Mobile_data cur_mobile = mobiles_data[mobile_ind];
		for(int feature_ind = 0 ; feature_ind < cur_mobile.features_cnt() ; feature_ind++){
			if(mobile_ind == 0){
				min_val.push_back(cur_mobile.feature[feature_ind]);
				max_val.push_back(cur_mobile.feature[feature_ind]);
				continue;
			}
			if(min_val[feature_ind] > cur_mobile.feature[feature_ind])
				min_val[feature_ind] = cur_mobile.feature[feature_ind];
			if(max_val[feature_ind] < cur_mobile.feature[feature_ind])
				max_val[feature_ind] = cur_mobile.feature[feature_ind];
		}
	}
}

void normalize_mobiles_data(Mobiles_data_list& mobiles_data){
	Feature_value_list min_val , max_val;
	find_min_and_max_value_of_each_feature(mobiles_data , min_val , max_val);
	for(int mobile_ind = 0 ; mobile_ind < mobiles_data.size() ; mobile_ind++){
		Mobile_data* cur_mobile = &mobiles_data[mobile_ind];
		for(int feature_ind = 0 ; feature_ind < cur_mobile->features_cnt() ; feature_ind++){
			cur_mobile->feature[feature_ind] -= min_val[feature_ind];
			cur_mobile->feature[feature_ind] /= (max_val[feature_ind] - min_val[feature_ind]);
		}
	}
}

void read_and_store_data_from_weight_file(ifstream& weight_file , Price_class_list& price_class_data){
	int feature_cnt = find_number_of_features(weight_file);
	string data_line;
	while(getline(weight_file , data_line)){
		stringstream ss(replace_comma_with_space(data_line));
		Price_class new_price_class;

		double tmp_weight;
		for(int feature_ind = 0 ; feature_ind < feature_cnt ; feature_ind++){
			ss >> tmp_weight;
			new_price_class.add_feature_weight(tmp_weight);
		}
		ss >> new_price_class.bios;

		price_class_data.push_back(new_price_class);
	}
}

void extract_data_from_trainfile(string dataset_path , Mobiles_data_list& mobiles_data){
	string train_file_path = get_train_file_path(dataset_path);
	ifstream train_file(train_file_path);
	read_and_store_data_from_train_file(train_file , mobiles_data);
	normalize_mobiles_data(mobiles_data);
	train_file.close();
}

void extract_weights_from_weight_file(string dataset_path , Price_class_list& price_class_data){
	string weight_file_path = get_weight_file_path(dataset_path);
	ifstream weight_file(weight_file_path);
	read_and_store_data_from_weight_file(weight_file , price_class_data);
	weight_file.close();
}

double calculate_accuracy(Mobiles_data_list mobiles_data , Price_class_list price_class_data){
	double correct = 0;
	for(int mobile_ind = 0 ; mobile_ind < mobiles_data.size() ; mobile_ind++){
		Mobile_data cur_mobile = mobiles_data[mobile_ind];
		int calculated_class = 0;
		double score = 0;
		for(int price_class_ind = 0 ; price_class_ind < price_class_data.size() ; price_class_ind++){
			Price_class cur_class = price_class_data[price_class_ind];
			double cur_class_score = cur_class.calculate_class_score(cur_mobile);
			calculated_class = (score < cur_class_score) ? price_class_ind : calculated_class;
			score = (score < cur_class_score) ? cur_class_score : score;
		}
		if(calculated_class == cur_mobile.type)
			correct += 1;
	}
	return (correct / mobiles_data.size()) * 100;
}


int main(int argc , char* argv[]){
	clock_t start1 , end1 , start2 , end2 , start3 , end3;
	if(argc != 2){
		cout << "Input Error!\n" << endl;
		return 0;
	}
	Mobiles_data_list mobiles_data;
	Price_class_list price_class_data;
	string dataset_path = argv[DATASET_IND];

	extract_data_from_trainfile(dataset_path , mobiles_data);
	extract_weights_from_weight_file(dataset_path , price_class_data);
	double accuracy = calculate_accuracy(mobiles_data , price_class_data);

	cout << fixed << setprecision(2) << "Accuracy: " << accuracy << PERCANTAGE << endl;
}