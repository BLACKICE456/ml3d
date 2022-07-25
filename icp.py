import numpy as np
import torch as torch
import torch.nn.functional as F
import sys
import os
sys.path.append("simpleICP-master/simpleicp")
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from simpleicp import PointCloud, SimpleICP


def test_one_epoch(pc):
    mse_ab = 0
    mae_ab = 0
    mse_ba = 0
    mae_ba = 0

    total_loss = 0
    total_cycle_loss = 0
    num_examples = 0
    rotations_ab = []
    translations_ab = []
    rotations_ab_pred = []
    translations_ab_pred = []

    rotations_ba = []
    translations_ba = []
    rotations_ba_pred = []
    translations_ba_pred = []

    eulers_ab = []
    eulers_ba = []
    for i in tqdm(range(len(pc))):
        
        num_examples += 1
        src, target, rotation_ab, translation_ab, rotation_ba, translation_ba, euler_ab, euler_ba = pc[i]
        rotation_ab = torch.tensor(rotation_ab).double()
        translation_ab = torch.tensor(translation_ab).double()
        rotation_ba = torch.tensor(rotation_ba).double()
        translation_ba = torch.tensor(translation_ba).double()
        euler_ab = torch.tensor(euler_ab).double()
        euler_ba = torch.tensor(euler_ba).double()


        src = PointCloud(src.T, columns=["x", "y", "z"])
        target = PointCloud(target.T, columns=["x", "y", "z"])
        # Create simpleICP object, add point clouds, and run algorithm!
        icp = SimpleICP()
        icp.add_point_clouds(target, src)
        H, X_mov_transformed, rigid_body_transformation_params = icp.run(max_overlap_distance=1)
        X_mov_transformed = torch.tensor(X_mov_transformed)

        # get the predicted rotations and translations
        rotation_ab_pred = H[:3, :3]
        translation_ab_pred = H[:3, 3]
        rotation_ab_pred = torch.tensor(rotation_ab_pred).double()
        translation_ab_pred = torch.tensor(translation_ab_pred).double()
        rotation_ba_pred = rotation_ab_pred.transpose(1, 0).contiguous()
        translation_ba_pred = -torch.matmul(rotation_ba_pred, translation_ab_pred.unsqueeze(1)).squeeze(1)

        ## save rotation and translation
        rotations_ab.append(rotation_ab.unsqueeze(0).detach().cpu().numpy())
        translations_ab.append(translation_ab.unsqueeze(0).detach().cpu().numpy())
        rotations_ab_pred.append(rotation_ab_pred.unsqueeze(0).detach().cpu().numpy())
        translations_ab_pred.append(translation_ab_pred.unsqueeze(0).detach().cpu().numpy())
        eulers_ab.append(euler_ab.unsqueeze(0).numpy())
        ##
        rotations_ba.append(rotation_ba.unsqueeze(0).detach().cpu().numpy())
        translations_ba.append(translation_ba.unsqueeze(0).detach().cpu().numpy())
        rotations_ba_pred.append(rotation_ba_pred.unsqueeze(0).detach().cpu().numpy())
        translations_ba_pred.append(translation_ba_pred.unsqueeze(0).detach().cpu().numpy())
        eulers_ba.append(euler_ba.unsqueeze(0).numpy())
        ##############################################
        identity = torch.eye(3).repeat(1, 1)
        loss = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(1, 0), rotation_ab), identity) \
                + F.mse_loss(translation_ab_pred, translation_ab)
        total_loss += loss.item()
        mse_ab += torch.mean((X_mov_transformed - torch.tensor(target.X)) ** 2, dim=[0, 1]).item()
        mae_ab += torch.mean(torch.abs(X_mov_transformed - torch.tensor(target.X)), dim=[0, 1]).item()

        transformed_target = transform(target.X, R=rotation_ba_pred, t=translation_ba_pred)
        mse_ba += torch.mean((transformed_target - torch.tensor(src.X)) ** 2, dim=[0, 1]).item()
        mae_ba += torch.mean(torch.abs(transformed_target - torch.tensor(src.X)), dim=[0, 1]).item()

    rotations_ab = np.concatenate(rotations_ab, axis=0)
    translations_ab = np.concatenate(translations_ab, axis=0)
    rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
    translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)

    rotations_ba = np.concatenate(rotations_ba, axis=0)
    translations_ba = np.concatenate(translations_ba, axis=0)
    rotations_ba_pred = np.concatenate(rotations_ba_pred, axis=0)
    translations_ba_pred = np.concatenate(translations_ba_pred, axis=0)

    eulers_ab = np.concatenate(eulers_ab, axis=0)
    eulers_ba = np.concatenate(eulers_ba, axis=0)

    return total_loss * 1.0 / num_examples, total_cycle_loss / num_examples, \
           mse_ab * 1.0 / num_examples, mae_ab * 1.0 / num_examples, \
           mse_ba * 1.0 / num_examples, mae_ba * 1.0 / num_examples, rotations_ab, \
           translations_ab, rotations_ab_pred, translations_ab_pred, rotations_ba, \
           translations_ba, rotations_ba_pred, translations_ba_pred, eulers_ab, eulers_ba
    
def test(dataset):
    test_loss, test_cycle_loss, \
    test_mse_ab, test_mae_ab, test_mse_ba, test_mae_ba, test_rotations_ab, test_translations_ab, \
    test_rotations_ab_pred, \
    test_translations_ab_pred, test_rotations_ba, test_translations_ba, test_rotations_ba_pred, \
    test_translations_ba_pred, test_eulers_ab, test_eulers_ba = test_one_epoch(dataset)
    test_rmse_ab = np.sqrt(test_mse_ab)
    test_rmse_ba = np.sqrt(test_mse_ba)

    test_rotations_ab_pred_euler = npmat2euler(test_rotations_ab_pred)
    print(test_rotations_ab_pred_euler)
    test_r_mse_ab = np.mean((test_rotations_ab_pred_euler - test_eulers_ab) ** 2)
    test_r_rmse_ab = np.sqrt(test_r_mse_ab)
    test_r_mae_ab = np.mean(np.abs(test_rotations_ab_pred_euler - test_eulers_ab))
    test_t_mse_ab = np.mean((test_translations_ab - test_translations_ab_pred) ** 2)
    test_t_rmse_ab = np.sqrt(test_t_mse_ab)
    test_t_mae_ab = np.mean(np.abs(test_translations_ab - test_translations_ab_pred))

    test_rotations_ba_pred_euler = npmat2euler(test_rotations_ba_pred, 'xyz')
    test_r_mse_ba = np.mean((test_rotations_ba_pred_euler - test_eulers_ba) ** 2)
    test_r_rmse_ba = np.sqrt(test_r_mse_ba)
    test_r_mae_ba = np.mean(np.abs(test_rotations_ba_pred_euler - test_eulers_ba))
    test_t_mse_ba = np.mean((test_translations_ba - test_translations_ba_pred) ** 2)
    test_t_rmse_ba = np.sqrt(test_t_mse_ba)
    test_t_mae_ba = np.mean(np.abs(test_translations_ba - test_translations_ba_pred))

    # filename = 'icp_matrix.txt'
    # with open(filename,'w') as f:
    #     f.write('==FINAL TEST==\n')
    #     f.write('A--------->B\n')
    #     f.write('EPOCH:: %d, Loss: %f, Cycle Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
    #                     'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f\n'
    #                     % (0, test_loss, test_cycle_loss, test_mse_ab, test_rmse_ab, test_mae_ab,
    #                         test_r_mse_ab, test_r_rmse_ab,
    #                         test_r_mae_ab, test_t_mse_ab, test_t_rmse_ab, test_t_mae_ab))
    #     f.write('B--------->A\n')
    #     f.write('EPOCH:: %d, Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
    #                     'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f\n'
    #                     % (0, test_loss, test_mse_ba, test_rmse_ba, test_mae_ba, test_r_mse_ba, test_r_rmse_ba,
    #                         test_r_mae_ba, test_t_mse_ba, test_t_rmse_ba, test_t_mae_ba))
    print('==FINAL TEST==')
    print('A--------->B')
    print('EPOCH:: %d, Loss: %f, Cycle Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
                    'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                    % (-1, test_loss, test_cycle_loss, test_mse_ab, test_rmse_ab, test_mae_ab,
                        test_r_mse_ab, test_r_rmse_ab,
                        test_r_mae_ab, test_t_mse_ab, test_t_rmse_ab, test_t_mae_ab))
    print('B--------->A')
    print('EPOCH:: %d, Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
                    'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                    % (-1, test_loss, test_mse_ba, test_rmse_ba, test_mae_ba, test_r_mse_ba, test_r_rmse_ba,
                        test_r_mae_ba, test_t_mse_ba, test_t_rmse_ba, test_t_mae_ba))
    return test_rotations_ab_pred, test_translations_ab_pred


def npmat2euler(mats, seq='zyx'):
    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_matrix(mats[i])
        eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype='float32')


def transform(point_cloud, R=None, t=None):
    t_broadcast = np.broadcast_to(t[:, np.newaxis], (3, point_cloud.shape[0]))
    return (R @ point_cloud.T + t_broadcast).T