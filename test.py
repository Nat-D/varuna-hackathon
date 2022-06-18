# Example on how to use the saved model with cpu

import torch
from model import *
from utils import *
from PIL import Image
import pandas as pd



def predictions(model, selected_days, test_data_dir, device):
    # selected pictures to predict
    days = selected_days

    for day in days:
        test_big_sample = np.load(test_data_dir + f"{day}.npy").astype(np.float32) #cpu only support 32 bits
        test_big_sample = np.expand_dims(test_big_sample, axis=0) # add a batch dimension [batch, height, width, channel]
        test_big_sample = np.transpose(test_big_sample, (0,3,1,2)) # data should be [batch, channel, height, width]

        with torch.no_grad():
            test_big_sample = torch.tensor(test_big_sample).to(device)
            pred = model(test_big_sample).squeeze()


        # take argmax & convert result to numpy  
        pred_max = torch.argmax(pred, dim=0, keepdim=False) 
        pred_np = pred_max.cpu().numpy()

    return pred_np

def test_prediction_csv(pred_np, test_mask_dir, output_dir):


    # get prediction in each polygon
    mask_dir = test_mask_dir
    all_mask = os.listdir(mask_dir)
    mask_pred_np = np.zeros((0, 2))
    for mask in all_mask:
        mask_np = np.load(mask_dir + mask).astype(np.float32)
        values, counts = np.unique(mask_np*pred_np, return_counts=True)
        sorted_index = sorted(range(len(counts)), key=lambda k: counts[k])

        max_value = values[sorted_index[-2]]
        poly_index = int(mask.replace(".npy", ""))

        mask_pred_np = np.vstack([mask_pred_np, np.array([poly_index, max_value]).astype(int)])

    sorted_mask_pred_np = mask_pred_np[mask_pred_np[:, 0].argsort()]

    # creating a list of column names
    column_values = ['index', 'crop_type']
    df = pd.DataFrame(data = sorted_mask_pred_np, columns = column_values).astype(int)
    df.to_csv(output_dir, index = False)  

if __name__ == "__main__":

    PATH = 'runs/3500/3500.tar'
    DEVICE = 'cpu'
    PREPROCESS = Standardize('PS', device= DEVICE)
    MODEL = NoNameUNET(in_channels=12, out_channels=5, preprocess=PREPROCESS).to(DEVICE)
    SELECTED_DAYS = [20210106]
    TEST_DATA_DIR = './test_data/'
    TEST_MASK_DIR = './test_output/mask/'
    OUTPUT_DIR = './output.csv'
    
    model = MODEL
    model.load_state_dict(torch.load(PATH, map_location = torch.device(DEVICE))["state_dict"])
    model.eval()  # model in testing model

    pred_np = predictions(model, SELECTED_DAYS, TEST_DATA_DIR, DEVICE)
    test_prediction_csv(pred_np, TEST_MASK_DIR, OUTPUT_DIR)