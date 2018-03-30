import configparser

Config = configparser.ConfigParser()

# lets create that config file for next time...
cfgfile = open("mangafacesconfig.ini",'w')

# add the settings to the structure of the file, and lets write it out...
Config.add_section('TFNet')
Config.set('TFNet','Model_Relative_Path', 'cfg/tiny-yolo-voc-1c.cfg')
Config.set('TFNet','Accuracy_Threshold', str(0.2))
Config.set('TFNet','Load_Min_Steps', str(750))
Config.set('TFNet','Load_Max_Steps', str(1750))
Config.set('TFNet','Load_Increment', str(250))
Config.set('TFNet','GPU_Usage', str(0.6))

Config.add_section('ValidationSet')
Config.set('ValidationSet','annotations_xml_path', 'validation_annotations/*.xml')

Config.add_section('IoU')
Config.set('IoU','IoU_Threshold', str(0.5))

Config.add_section('ConfusionMatrix')
Config.set('ConfusionMatrix','weight_matrix_path', 'weight_matrices/weight_matrix_labels_9.txt')
Config.write(cfgfile)
cfgfile.close()