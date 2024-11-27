from os import environ

#  turning off redunndant warnings from tensorflow
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print("TF_CPP_MIN_LOG_LEVEL set to ", environ['TF_CPP_MIN_LOG_LEVEL'])

# turnning off onednn from tensorflow
environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
print("TF_ENABLE_ONEDNN_OPTS set to ", environ['TF_ENABLE_ONEDNN_OPTS'])
