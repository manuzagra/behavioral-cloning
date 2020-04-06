import pathlib
import csv


def get_data_info(path):
    parent = pathlib.Path(path)
    
    samples = []
    for csv_file in parent.glob('**/*.csv'):
        with open(str(csv_file), 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                try:
                    samples.append({'img_center': str(csv_file.resolve().parent.joinpath('IMG',pathlib.Path(line[0]).name)),
                                    'img_left': str(csv_file.resolve().parent.joinpath('IMG',pathlib.Path(line[1]).name)),
                                    'img_right': str(csv_file.resolve().parent.joinpath('IMG',pathlib.Path(line[2]).name)),
                                    'steering_angle': float(line[3]),
                                    'throttle': float(line[4]),
                                    'brake': float(line[5]),
                                    'speed': float(line[6])})
                except Exception as e:
                    print(e)
    return samples


def get_field_as_list(data, field):
    return [x[field] for x in data]
    
                      
if __name__ == '__main__':

    import utils
    utils.save_data(get_data_info('/opt/sim_data'), './images_data.p')