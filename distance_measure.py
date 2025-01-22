import pandas as pd
from pathlib import Path
import numpy as np

# Function to calculate distances and handle missing tracked_object_ids
def calculate_distances(df, source, video_size, save_csv = False):
    p = Path(source)

    video_diagonal = np.sqrt(video_size**2 + video_size**2)

    # Add center_x and center_y columns
    df['center_x'] = (df['x1'] + df['x2']) / 2
    df['center_y'] = (df['y1'] + df['y2']) / 2

    # Separate class 0 and class 1 objects
    class_0 = df[df['cls'] == 0]
    
    class_1 = df[df['cls'] == 1]
    

    # Prepare a dictionary to store distances
    distance_dict = {}

    # Get all unique frames
    all_frames = sorted(df['frame'].unique())
    

    #Iterate through each class 1 object and calculate distances
    for obj_id, obj_data in class_1.groupby('tID'):
        distances = []
        for frame in all_frames:
            class_0_frame = class_0[class_0['frame'] == frame]
            class_1_frame = obj_data[obj_data['frame'] == frame]
            if not class_1_frame.empty and not class_0_frame.empty:
                x1, y1 = class_1_frame.iloc[0][['center_x', 'center_y']]
                distance = class_0_frame.apply(
                    lambda row: np.sqrt((x1 - row['center_x'])**2 + (y1 - row['center_y'])**2),
                    axis=1
                ).min()
                norm_dist = distance/video_diagonal
                distances.append(norm_dist)
            elif not class_1_frame.empty:  # Class 1 object is present but no Class 0
                distances.append(np.nan)
            else:  # Class 1 object is not present in this frame
                distances.append("NA")
        distance_dict[obj_id] = distances

    # Create output DataFrame
    output_df = pd.DataFrame(distance_dict, index=all_frames)
    output_df.index.name = 'frame'
    output_df = output_df.transpose()  # Rows as objects, columns as frames
    return output_df

    # Save to CSV
    if save_csv:
        output_df.to_csv(f'{p.stem}_distances.csv')

# Example usage
# Replace this with your DataFrame loading code
# df = pd.read_csv('your_input_file.csv')
# calculate_distances(df)
