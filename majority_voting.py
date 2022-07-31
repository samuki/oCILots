import argparse




def perform_post_processing(data):

    for datapoint in data.keys():
        if data[datapoint] == 0:
            continue

        image_number = datapoint.split("_")[0]
        point_x = datapoint.split("_")[1]
        point_y = datapoint.split("_")[2]

        if int(point_x) == 0:
            neighbour_left = 0
        else: 
            key = image_number + "_" + str(int(point_x) - 16) + "_" + str(int(point_y))
            neighbour_left = data[key]

        if int(point_x) == 384:
            neighbour_right = 0
        else:
            key = image_number + "_" + str(int(point_x) + 16) + "_" + str(int(point_y))
            neighbour_right = data[key]

        if int(point_y) == 0:
            neighbour_top = 0
        else:
            key = image_number + "_" + str(int(point_x)) + "_" + str(int(point_y)-16)
            neighbour_top = data[key]

        if int(point_y) == 384:
            neighbour_bottom = 0
        else:
            key = image_number + "_" + str(int(point_x)) + "_" + str(int(point_y)+16)
            neighbour_bottom = data[key]

        if neighbour_bottom == 0 and neighbour_left == 0 and neighbour_right == 0 and neighbour_top == 0:
            data[datapoint] = 0

    return data





def perform_majority_voting(files): 

    data = []

    for file in files:
        with open(file, "r") as f:
            results = {elm.split(",")[0] : elm.split(",")[1].replace("\n", "") for elm in f.readlines()}
            data.append(results)
    
    base_data = data[0]

    for key in base_data.keys():
        
        if key == "id":
            continue
        
        points = []
        for item in data:
            points.append(int(item[key]))
        
        if sum(points) >= 2:
            base_data[key] = 1
        else:
            base_data[key] = 0

    return base_data



if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description="Please provide submission csv on which to perform majority voting",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", nargs='*', required=True)

    args = parser.parse_args()
    result =   perform_majority_voting(args.i)

    with open("majority_voting_result.csv", "w") as f:
        f.write("id,prediction\n")
        for key in result.keys():
            f.write(key + "," + str(result[key]) + "\n")