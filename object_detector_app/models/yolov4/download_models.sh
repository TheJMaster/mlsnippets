#!/bin/bash

wget --no-check-certificate "https://locteststorage3.blob.core.windows.net/privnet/lohuynh/Models/yolov4_320_norm.pb?sp=rl&st=2020-06-12T05:36:10Z&se=2021-06-13T05:36:00Z&sv=2019-10-10&sr=b&sig=2C%2FzXrg1SrxWp80WRZB0bowSJRpxqkFQSAT2cUjZfVI%3D" -O ./models/yolov4_320_norm.pb

wget --no-check-certificate "https://locteststorage3.blob.core.windows.net/privnet/lohuynh/Models/yolov4_416_norm.pb?sp=rl&st=2020-06-12T05:37:17Z&se=2021-06-13T05:37:00Z&sv=2019-10-10&sr=b&sig=pf49GZdqVSYULvYJ9hxwZHF3KyQE181nuiCnmmVRTXY%3D" -O ./models/yolov4_416_norm.pb

wget --no-check-certificate "https://locteststorage3.blob.core.windows.net/privnet/lohuynh/Models/yolov4_512_norm.pb?sp=rl&st=2020-06-12T05:37:35Z&se=2021-06-13T05:37:00Z&sv=2019-10-10&sr=b&sig=CNy2Q%2FKPAAClM26n5YXNWiXEjip2Gg8covcONl3m8kM%3D" -O ./models/yolov4_512_norm.pb

wget --no-check-certificate "https://locteststorage3.blob.core.windows.net/privnet/lohuynh/Models/yolov4_608_norm.pb?sp=rl&st=2020-06-12T05:37:55Z&se=2021-06-13T05:37:00Z&sv=2019-10-10&sr=b&sig=xW9PFS5weqho6JPBnMNQ7NVYVMGcioOvRCqri2dl9Tc%3D" -O ./models/yolov4_608_norm.pb