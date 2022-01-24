import os

label_map_dict = {}


def generate_labels(dirss):

    labels_map = open('labels_map.txt', 'w')
    labels = open('labels.txt', 'w')

    list_labels = []

    print('Starting walk!')
    for dir in dirss:
        for subdir, dirs, files in os.walk(dir):
            for i, dir in enumerate(dirs):
                if dir not in list_labels:
                    list_labels.append(dir)
                    print(i, len(dirs))
            break
    print('Walk DONE!')
    print('Starting walk!')

    labels_sorted=sorted(list_labels)

    for i, item in enumerate(labels_sorted):
        labels_map.write(str(item) + ':' + str(i) +'\n')
        label_map_dict[str(item)] = str(i)

        labels.write(str(i) + '\n')

    print(label_map_dict)


def generate_train_list_file(paths):

    import random

    class_files = {}
    list_of_train_data = []
    train_list = open('train_list_CGIAR.txt', 'w')

    for i_path, path in enumerate(paths):
        print(i_path)
        for subdir, dirs, files in os.walk(path):
            label = subdir.split('/')[-1]
            print(label)
            if label == '':
                continue

            files = [os.path.join(subdir, file) + ' ' + label_map_dict[label] for file in files]
            if label not in class_files:
                if label is not '':
                    class_files[label] = files
            else:
                class_files[label].extend(files)

    for _, files in class_files.items():

        random.shuffle(files)
        random.shuffle(files)

        list_of_train_data.extend(files)

    # shuffle twice for no reason
    random.shuffle(list_of_train_data)
    random.shuffle(list_of_train_data)

    for image_path in list_of_train_data:
        train_list.write(image_path + '\n')


def generate_test_list_file(path):

    images = []

    test_list = open('test_list_CGIAR.txt', 'w')

    for subdir, dirs, files in os.walk(path[0]):

        for file in files:
            file_path = os.path.join(subdir, file)

            images.append(file_path)

    for image_path in images:

        test_list.write(image_path + ' ' + '0' +'\n')



generate_labels(['train_images/'])
print("Labels generated.")

generate_test_list_file(['test_images/'])
print("Test list file generated.")

generate_train_list_file(['train_images/'])
print("Train list file generated.")

