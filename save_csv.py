import csv


def write_to_csv(img_info):

    with open('brightness_new.csv', 'w') as csvfFile:
        fields = [
            'ID', 'image_name', 'left_brightness', 'right_brightness',
            'annotation'
        ]
        writer = csv.writer(csvfFile)
        writer.writerow(fields)
        for i, row in enumerate(img_info):
            row.insert(0, str(i))
            writer.writerow(row)
