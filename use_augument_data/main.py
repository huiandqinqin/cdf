from use_augument_data.my_utils.get_train_test import get_train_test_1dim
import json
import ast


str = "{'cnn': {'0HP': 10.429042726755142, '1HP': 11.155115514993668, '2HP': 10.894039794802666, '3HP': 9.407894611358643}, 'cnnDF': {'0HP': 99.10891089108911, '1HP': 96.8976897689769, '2HP': 99.63576158940397, '3HP': 99.96710526315789}, 'deep_forest': {'0HP': 84.62046204620461, '1HP': 78.77887788778878, '2HP': 80.33112582781457, '3HP': 85.88815789473684}, 'wdcnn': {'0HP': 19.570956826210022, '1HP': 15.874587595462799, '2HP': 18.774834349751472, '3HP': 18.28947350382805}, 'wdcnnDF': {'0HP': 97.42574257425743, '1HP': 97.95379537953795, '2HP': 98.80794701986756, '3HP': 97.79605263157895}}"
test_string = '{"Nikhil" : 1, "Akshat" : 2, "Akash" : 3}'

list = json.loads(test_string)
list2 = ast.literal_eval(str)
print(list2)
print(type(list2))