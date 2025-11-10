'''
This is the main file for the pathoptim package. Given an ensemble of realizations of the latent vector, the pathfinder
class will make realizations of the earthmodel by running the GAN and use there realizations to find the optimal path
for the well.
'''

from .DP import process_prior_and_plot_results, perform_dynamic_programming
from tqdm import tqdm
import multiprocessing
import numpy as np

# todo rename to capital Pathfinder
class pathfinder():
    #def __init__(self):
    # def apply_simulation(self, args):
    #     state, best_pos, index = args
    #     return self.sim.run_fwd_sim(state, best_pos, index)
    #
    # def worker(self,state_index):
    #     state, index = state_index
    #     return self.sim.run_fwd_sim(state, index)
    #
    # def parallel_process(self,states, indices, num_cpus):
    #     with multiprocessing.Pool(processes=num_cpus) as pool:
    #         # Map the worker to the data
    #         results = list(tqdm(pool.imap(self.worker, zip(states, indices)), total=len(states), disable=self.disable_tqdm))
    #     return results

    def trace_path(self, dp_matrix, start_row, start_col):
        path = [start_row]  # Start the path with the best index
        current_row = start_row
        # Trace the path forward from the start column to the end of the matrix
        for col in range(start_col, dp_matrix.shape[1]):
            # Assuming the next row is determined by the maximum value in the next column of the current row
            # Adjust this as necessary based on your DP formulation
            current_row = np.argmax(dp_matrix[:, col])
            path.append(current_row)

        return path

    def get_cost_vector(self, dy_vector=None, cost_mult=0.02):
        """
        Computes drilling cost as the euclidian distance using the preset cell dimensions
        :param dy_vector:
        :param cost_mult:
        :return:
        """
        if dy_vector is None:
            dy_vector = np.array([0, -1, 1])
        # 10 in dx 0.5 in dy
        cost_vector = np.array(np.sqrt(np.power(10.0, 2) + np.power(dy_vector * 0.5, 2))) * cost_mult
        return cost_vector

    def no_gan_run(self,
                   weighted_images,
                   start_point,
                   discount_for_remainder=1.0,
                   dy_vector=None,
                   cost_vector=None,
                   cost_mult=0.02,
                   recompute_optimal_paths_from_next=False
                   ):
        """
        :param weighted_images: A torch tensor representing the ensemble of value-images of shape Ne,X,Y
        :param start_point:
        :param discount_for_remainder:
        :param dy_vector:
        :param cost_vector:
        :param cost_mult:
        :return:
        """
        if dy_vector is None:
            dy_vector = np.array([0, -1, 1])
        if cost_vector is None:
            cost_vector = self.get_cost_vector(dy_vector, cost_mult)

        optimal_paths = []
        max_path_values = []
        dp_matrices = []
        ne = weighted_images.shape[0]

        cur_row, cur_column = start_point
        list_member_index = list(range(ne))

        for i in range(ne):
            weighted_image_i_np = weighted_images[i,:,:].to("cpu").numpy()
            dp_matrix, max_path_value, optimal_path = perform_dynamic_programming(weighted_image_i_np,
                                                                                  start_point,
                                                                                  di_vector=dy_vector,
                                                                                  cost_vector=cost_vector)

            optimal_paths.append(optimal_path)
            max_path_values.append(max_path_value)
            dp_matrices.append(dp_matrix)

        dp_matrix_shape = dp_matrices[0].shape

        next_column = cur_column + 1
        best_next_index = None
        if next_column < dp_matrix_shape[1]:
            # initialize all as unreachable
            sum_for_column = np.ones(dp_matrix_shape[0]) * -1.0
            # iterate over rows
            for k, dy in enumerate(dy_vector):
                y = cur_row + dy
                drilling_direction_cost = cost_vector[k]
                # sum over ensemble members
                for i in range(ne):
                    # we add the expected immidiate reward and the expected long-term gain times the discount
                    sum_for_column[y] += (weighted_images[i][y][next_column]
                                      + dp_matrices[i][y][next_column] * discount_for_remainder)
            if np.max(sum_for_column) > 0:
                best_next_index = np.argmax(sum_for_column)

        next_best_position = (best_next_index, next_column)
        # list_best_pos = [next_best_position[0]] * ne

        if recompute_optimal_paths_from_next:
            optimal_paths_remainder = []
            for i in range(ne):
                # todo recover this path from matrix for effciency
                weighted_image_i_np = weighted_images[i, :, :].to("cpu").numpy()
                dp_matrix, max_path_value, optimal_path = perform_dynamic_programming(weighted_image_i_np,
                                                                                      next_best_position,
                                                                                      di_vector=dy_vector,
                                                                                      cost_vector=cost_vector)
                optimal_paths_remainder.append(optimal_path)
            return next_best_position, optimal_paths_remainder
        else:
            return next_best_position, optimal_paths


    def run(self,
            state,
            start_point,
            gan_evaluator,
            discount_for_remainder=1.0,
            dy_vector=None,
            cost_vector=None,
            cost_mult=0.02):
        if dy_vector is None:
            dy_vector = np.array([0, -1, 1])
        if cost_vector is None:
            cost_vector = self.get_cost_vector(dy_vector, cost_mult)

        optimal_paths = []
        max_path_values = []
        dp_matrices = []
        weighted_images = []

        ne = state.shape[1]
        # change it to an input parameter
        # row first, column second
        #start_point = (32, 0)  # Middle of the image
        cur_row, cur_column = start_point

        list_member_index = list(range(ne))
        for i in range(ne):

            # todo send the di vector there
            dp_matrix_i, max_path_value, weighted_image_i, optimal_path = \
                process_prior_and_plot_results(state[:, i],
                                               start_point,
                                               gan_evaluator,
                                               di_vector=dy_vector,
                                               cost_vector=cost_vector)

            optimal_paths.append(optimal_path)
            max_path_values.append(max_path_value)
            dp_matrices.append(dp_matrix_i)
            weighted_images.append(weighted_image_i)

        next_column = cur_column + 1
        best_next_index = None
        if next_column < dp_matrix_i.shape[1]:
            # initialize all as unreachable
            sum_for_column = np.ones(dp_matrix_i.shape[0])*-1
            # iterate over rows
            # for y in range(dp_matrix_i.shape[0]):
            for k, dy in enumerate(dy_vector):
                y = cur_row + dy
                drilling_direction_cost = cost_vector[k]
                # sum over ensemble members
                for i in range(ne):
                    # we add the expected immidiate reward and the expected long-term gain times the discount
                    sum_for_column[y] += (weighted_images[i][y][next_column]
                                          # - drilling_direction_cost
                                          + dp_matrices[i][y][next_column] * discount_for_remainder)
            if np.max(sum_for_column) > 0:
                best_next_index = np.argmax(sum_for_column)
        #else:
        #    print("We are done")

        next_best_position = (best_next_index, next_column)
        list_best_pos = [next_best_position[0]] * ne
        # Run prediction in parallel using p_map
        # Map function over paired states and indices
        #results = list(
        #    tqdm(map(self.apply_simulation, zip(self.ensemble, list_best_pos, list_member_index)), total=len(self.ensemble),
        #         disable=self.disable_tqdm))
        #en_pred = results

        return next_best_position, list_best_pos
