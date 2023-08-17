import numpy as np
from scipy.sparse import find
from collections import defaultdict


class DecisionTreeRecArtistModel:
    def __init__(self, source_matrix, artist_matrix, top_items, top_artists,
                 tree_depth_threshold=6, lambda_value=7,
                 top_items_num=1000, top_artists_num=50):
        self.tree_depth_threshold = tree_depth_threshold
        self.lambda_value = lambda_value
        self.current_depth = 0
        self.top_items_num = top_items_num
        self.top_items_list = top_items
        self.real_items_num = source_matrix.shape[1]
        input_data = find(source_matrix)
        users_list = input_data[0]
        items_list = input_data[1]
        self.rU = {}
        self.global_mean = 0  # global average of ratings

        # artist related data
        self.top_artists_num = top_artists_num
        self.top_artists_list = top_artists
        self.real_artists_num = artist_matrix.shape[1]
        artist_data = find(artist_matrix)
        artist_users_list = artist_data[0]
        artist_list = artist_data[1]
        self.rG = {}
        # self.track_artist_mapping = track_artist_dict

        # Calculate rate of progress
        self.node_num = 0
        self.current_node_t = 0
        for i in range(self.tree_depth_threshold):
            self.node_num += 3 ** i

        # Generate rI, rU item and user data
        print("Generate the item list and user interactions dictionary")
        self.rI = list(set(items_list))
        for user_id, item_id in zip(users_list, items_list):
            rating = source_matrix[user_id, item_id]
            self.rU.setdefault(user_id, {})[item_id] = rating
            self.global_mean += rating
        self.global_mean /= len(items_list)
        self.item_size = len(self.rI)
        self.user_size = len(self.rU)

        # Generate rArtists, rG artist data
        print("Generate the artist interactions dictionary")
        self.rArtists = list(set(artist_list))
        # self.sum_artist_ratings = 0
        # self.sum_square_artist_ratings = 0
        for user_id, artist_id in zip(artist_users_list, artist_list):
            # if user_id in self.rU:
            rating = artist_matrix[user_id, artist_id]
            self.rG.setdefault(user_id, {})[artist_id] = rating
            # self.sum_artist_ratings += rating
            # self.sum_square_artist_ratings += rating ** 2

        '''        
        self.user_item_artist_rating = {}
        self.user_item_artist_average_rating = {}
        self.rU_item_average_rating = {}
        for user_id, items in self.rU.items():
            for item_id, item_rating in items.items():
                artist_ratings = []
                artist_id_list = self.track_artist_mapping.get(item_id, [])
                for artist_id in artist_id_list:
                    artist_rating = self.rG.get(user_id, {}).get(artist_id)
                    if artist_rating:
                        self.user_item_artist_rating.setdefault(user_id, {}).setdefault(item_id, {})[artist_id] = artist_rating
                        artist_ratings.append(artist_rating)
                if len(artist_ratings) > 0:
                    artist_average_rating = np.mean(artist_ratings)
                    self.user_item_artist_average_rating.setdefault(user_id, {})[item_id] = artist_average_rating
                    self.rU_item_average_rating.setdefault(user_id, {})[item_id] = (item_rating + artist_average_rating)/2
                else:
                    self.rU_item_average_rating.setdefault(user_id, {})[item_id] = item_rating           

        '''
        print("Generation Complete. Initialize tree params")
        # Initiate Tree, users_bound
        self.tree = list(self.rU.keys())
        self.split_items_list = []
        self.users_bound = {'0': [[0, len(self.tree) - 1]]}
        print("Tree params initialized. Populate error related stats")

        # Generate bias, sum_cur_node_t, sum_cur_square_node_t
        self.biasU = {}
        self.sum_cur_node_t = np.zeros(self.real_items_num)
        self.sum_cur_square_node_t = np.zeros(self.real_items_num)
        self.cnt_cur_node_t = np.zeros(self.real_items_num)
        self.sum_cur_node_t_artist = np.zeros(self.real_artists_num)
        self.sum_cur_square_node_t_artist = np.zeros(self.real_artists_num)
        self.cnt_cur_node_t_artist = np.zeros(self.real_artists_num)
        self.populate_stats_node_t()

        # Prediction Model
        self.pred_profile = {}
        print("Tree Initialized. Building Tree")

    def populate_stats_node_t(self):
        for user_id in self.rU:
            self.biasU[user_id] = (sum(list(self.rU[user_id].values())) + self.lambda_value * self.global_mean) / (
                    self.lambda_value + len(self.rU[user_id]))
            user_all_items_rating_id = np.array(list(self.rU[user_id].keys()))
            user_all_rating = np.array(list(self.rU[user_id].values()))
            self.sum_cur_node_t[user_all_items_rating_id[:]] += user_all_rating[:]  # - self.biasU[user_id]
            self.sum_cur_square_node_t[user_all_items_rating_id[:]] += (
                                                                       user_all_rating[:]) ** 2  # - self.biasU[user_id]
            self.cnt_cur_node_t[user_all_items_rating_id[:]] += 1
        for user_id in self.rG:
            user_all_items_rating_id_artist = np.array(list(self.rG[user_id].keys()))
            user_all_rating_artist = np.array(list(self.rG[user_id].values()))
            self.sum_cur_node_t_artist[user_all_items_rating_id_artist[:]] += user_all_rating_artist[
                                                                            :]  # - self.biasU[user_id]
            self.sum_cur_square_node_t_artist[user_all_items_rating_id_artist[:]] += (user_all_rating_artist[
                                                                                    :]) ** 2  # - self.biasU[user_id]
            self.cnt_cur_node_t_artist[user_all_items_rating_id_artist[:]] += 1
        # self.sum_cur_node_t += self.sum_artist_ratings
        # self.sum_cur_square_node_t += self.sum_square_artist_ratings

    def generate_decision_tree(self, users_bound_for_node, chosen_item_id):
        # print("Build tree function start")
        # print("users_bound_for_node", users_bound_for_node)
        # print("self.tree", self.tree)
        # Terminate
        self.current_depth += 1
        # print("self.current_depth > self.tree_depth_threshold",self.current_depth > self.tree_depth_threshold)
        # print("len(chosen_item_id) == self.item_size",len(chosen_item_id) == self.item_size)

        if self.current_depth > self.tree_depth_threshold or len(chosen_item_id) == self.item_size:
            # print("TERMINATE")
            return

        # Choose popular rated items
        # num_rec = np.zeros(self.real_items_num)
        # for user_id in self.tree[users_bound_for_node[0]:(users_bound_for_node[1] + 1)]:
        #    user_all_items_rating_id = np.array(list(self.rU[user_id].keys()))
        #    num_rec[user_all_items_rating_id[:]] += 1
        # sub_item_id = np.argsort(num_rec)[::-1][:self.top_items_num]
        # print("Items loaded")

        # Choose popular rated items
        # num_rec = np.zeros(self.real_artists_num)
        # for user_id in self.tree[users_bound_for_node[0]:(users_bound_for_node[1] + 1)]:
        #    if user_id in self.rG:
        #        user_all_artists_rating_id = np.array(list(self.rG[user_id].keys()))
        #        num_rec[user_all_artists_rating_id[:]] += 1
        # sub_artist_id = np.argsort(num_rec)[::-1][:self.top_artists_num]
        sub_item_id = self.rI  # np.array(self.top_items_list)
        sub_artist_id = self.rArtists  # np.array(self.top_artists_list)

        # if len(self.tree[users_bound_for_node[0]:(users_bound_for_node[1] + 1)]) > 0:
        #    print("NUMBER USER: ", len(self.tree[users_bound_for_node[0]:(users_bound_for_node[1] + 1)]))
        # print("num sub_item_id: ", len(sub_item_id))
        # print("num sub_artist_id: ", len(sub_artist_id))

        # print("Items loaded")
        # print("start error")
        # Find optimum item to split
        min_cnt_node_t, min_sum_node_t, min_sum_square_node_t, min_cnt_node_t_artist, \
            min_sum_node_t_artist, min_sum_square_node_t_artist, optimum_item_id, optimum_artist_id \
            = self.populate_items_error_list(chosen_item_id, sub_item_id, sub_artist_id, users_bound_for_node)
        # print("items_error_list calculation complete")
        # Find optimum split-item
        # print("end error")
        # filtered_errors = {key: value for key, value in items_error_list.items() if value != 0}
        # Check if there are any values greater than zero
        # if filtered_errors:
        # Get the key with the minimum value greater than zero
        # optimum_item_id = min(filtered_errors, key=filtered_errors.get)
        # else:
        # print("optimum_itemid NOT KNOWN")
        # optimum_item_id = min(items_error_list, key=items_error_list.get)

        # print("optimum_itemid", optimum_item_id, "Error: ", items_error_list[optimum_item_id])

        if len(self.split_items_list) == self.current_depth - 1:
            self.split_items_list.append([optimum_item_id])
        else:
            self.split_items_list[self.current_depth - 1].append(optimum_item_id)
        # self.split_items_list.setdefault(str(self.current_depth-1), []).append(optimum_item_id)
        chosen_item_id.append(optimum_item_id)

        # sort tree
        self.users_bound.setdefault(str(self.current_depth), []).append([])  # for LIKE
        self.users_bound[str(self.current_depth)].append([])  # for DISLIKE
        self.users_bound[str(self.current_depth)].append([])  # for UNKNOWN
        list_unknown, list_like, list_dislike = [], [], []
        # print("optimum_item_id", optimum_item_id)
        # print("split_items_list", self.split_items_list[self.current_depth-1])
        for user_id in self.tree[users_bound_for_node[0]:(users_bound_for_node[1] + 1)]:
            artist_rating_dict_values = self.rG.get(user_id, {}).get(optimum_artist_id)
            artist_rating = artist_rating_dict_values if artist_rating_dict_values is not None else None
            # print("user_id",user_id)
            if artist_rating is None or optimum_item_id not in self.rU[user_id] or self.rU[user_id][optimum_item_id] < 0:
                list_unknown.append(user_id)
            elif self.rU[user_id][optimum_item_id] > 50 and artist_rating > 50:
                list_like.append(user_id)
            elif self.rU[user_id][optimum_item_id] <= 50 and artist_rating <= 50:
                list_dislike.append(user_id)
            else:
                list_unknown.append(user_id)

        self.tree[users_bound_for_node[0]:(users_bound_for_node[1] + 1)] = list_like + list_dislike + list_unknown
        self.users_bound[str(self.current_depth)][-3] = [users_bound_for_node[0],
                                                         users_bound_for_node[0] + len(list_like) - 1]  # for LIKE
        self.users_bound[str(self.current_depth)][-2] = [users_bound_for_node[0] + len(list_like),
                                                         users_bound_for_node[0] + len(list_like) + len(
                                                             list_dislike) - 1]  # for DISLIKE
        self.users_bound[str(self.current_depth)][-1] = [users_bound_for_node[0] + len(list_like) + len(list_dislike),
                                                         users_bound_for_node[0] + len(list_like) + len(
                                                             list_dislike) + len(
                                                             list_unknown) - 1]  # for UNKNOWN
        # print("self.tree inside:", self.tree[users_bound_for_node[0]:(users_bound_for_node[1]+1)] )
        # print("list_like + list_dislike + list_unknown", list_like , list_dislike , list_unknown)
        # print("self.current_depth", str(self.current_depth))
        # print("self.users_bound[str(self.current_depth)]", self.users_bound[str(self.current_depth)])
        # print("self.users_bound", self.users_bound)

        # Generate Subtree of Node LIKE
        # print("Generate Subtree of Node LIKE")
        #level1 = str(self.current_depth)
        # print("level:", level1)
        # print("self.split_items_list[level-1]", self.split_items_list[self.current_depth-1])
        # print("users_bound[level]:", self.users_bound[level1], ", range(len(users_bound[level]):", range(len(self.users_bound[level1])))
        self.sum_cur_node_t = min_sum_node_t[:, 0]
        self.sum_cur_square_node_t = min_sum_square_node_t[:, 0]
        self.cnt_cur_node_t = min_cnt_node_t[:, 0]
        self.sum_cur_node_t_artist = min_sum_node_t_artist[:, 0]
        self.sum_cur_square_node_t_artist = min_sum_square_node_t_artist[:, 0]
        self.cnt_cur_node_t_artist = min_cnt_node_t_artist[:, 0]
        self.generate_decision_tree(self.users_bound[str(self.current_depth)][-3], chosen_item_id[:])
        self.current_depth -= 1

        # Generate Subtree of Node DISLIKE
        # print("Generate Subtree of Node DISLIKE")
        self.sum_cur_node_t = min_sum_node_t[:, 1]
        self.sum_cur_square_node_t = min_sum_square_node_t[:, 1]
        self.cnt_cur_node_t = min_cnt_node_t[:, 1]
        self.sum_cur_node_t_artist = min_sum_node_t_artist[:, 1]
        self.sum_cur_square_node_t_artist = min_sum_square_node_t_artist[:, 1]
        self.cnt_cur_node_t_artist = min_cnt_node_t_artist[:, 1]
        self.generate_decision_tree(self.users_bound[str(self.current_depth)][-2], chosen_item_id[:])
        self.current_depth -= 1

        # Generate Subtree of Node UNKNOWN
        # print("Generate Subtree of Node UNKNOWN")
        self.sum_cur_node_t = min_sum_node_t[:, 2]
        self.sum_cur_square_node_t = min_sum_square_node_t[:, 2]
        self.cnt_cur_node_t = min_cnt_node_t[:, 2]
        self.sum_cur_node_t_artist = min_sum_node_t_artist[:, 2]
        self.sum_cur_square_node_t_artist = min_sum_square_node_t_artist[:, 2]
        self.cnt_cur_node_t_artist = min_cnt_node_t_artist[:, 2]
        self.generate_decision_tree(self.users_bound[str(self.current_depth)][-1], chosen_item_id[:])
        self.current_depth -= 1
        '''
        # Show Rating Progress
        for i in range(self.current_depth - 1):
            print("┃", end="")
        print("┏", end="")
        self.current_node_t += 1
        print("Current depth: " + str(self.current_depth) + "        %.2f%%" % (100 * self.current_node_t / self.node_num))
        '''

    def populate_items_error_list(self, chosen_item_id, sub_item_id, sub_artist_id, users_bound_for_node):
        # items_error_list = {}
        min_error_item = float('inf')
        min_artist_id = -1
        min_item_id = -1
        node_users = self.tree[users_bound_for_node[0]:(users_bound_for_node[1] + 1)]
        min_sum_node_t = np.zeros((self.real_items_num, 3))
        min_sum_square_node_t = np.zeros((self.real_items_num, 3))
        min_cnt_node_t = np.zeros((self.real_items_num, 3))
        min_sum_node_t_artist = np.zeros((self.real_artists_num, 3))
        min_sum_square_node_t_artist = np.zeros((self.real_artists_num, 3))
        min_cnt_node_t_artist = np.zeros((self.real_artists_num, 3))

        if len(node_users) > 0:
            for item_id in set(sub_item_id) - set(chosen_item_id):
                ''' 
                    user_item_rating_node_t: [ [uid01, rating01], [uid02, rating02], ... ] 
                    to find all users in node t who rates item i
                '''
                # print("Item: ",  item_id)
                user_item_rating_node_t = [[user_id, self.rU[user_id][item_id]] for user_id in node_users if
                                           item_id in self.rU[user_id]]
                for artist_id in sub_artist_id:
                    # print ("Artist: ", artist_id)
                    sum_node_t = np.zeros((self.real_items_num, 3))
                    sum_square_node_t = np.zeros((self.real_items_num, 3))
                    cnt_node_t = np.zeros((self.real_items_num, 3))
                    sum_node_t_artist = np.zeros((self.real_artists_num, 3))
                    sum_square_node_t_artist = np.zeros((self.real_artists_num, 3))
                    cnt_node_t_artist = np.zeros((self.real_artists_num, 3))
                    for user_rating in user_item_rating_node_t:
                        # print("User rating: ", user_rating)
                        ''' user_all_items_rating_id: array [ [item_id11, rating11], [item_id12, rating12], ... ] '''
                        user_all_items_rating_id = np.array(list(self.rU[user_rating[0]].keys()))
                        user_all_rating = np.array(list(self.rU[user_rating[0]].values()))

                        artist_rating_dict_values = self.rG.get(user_rating[0], {}).get(artist_id)
                        artist_rating = artist_rating_dict_values if artist_rating_dict_values is not None else None

                        if artist_rating:
                            # print("User Rating: ", user_rating[1],"Artist Rating: ", artist_rating)
                            user_all_items_rating_id_artist = np.array(list(self.rG[user_rating[0]].keys()))
                            user_all_rating_artist = np.array(list(self.rG[user_rating[0]].values()))

                            valid_ratings_like = np.logical_and(user_rating[1] > 50, artist_rating > 50)
                            valid_ratings_dislike = np.logical_and(
                                np.logical_and(user_rating[1] >= 0, user_rating[1] <= 50),
                                np.logical_and(artist_rating >= 0, artist_rating <= 50))
                            # print ("Like: ", valid_ratings_like, ", Dislike: ", valid_ratings_dislike)
                            # calculate sum_tL for node LIKE
                            if valid_ratings_like:
                                sum_node_t[user_all_items_rating_id[:], 0] += user_all_rating[:]
                                sum_square_node_t[user_all_items_rating_id[:], 0] += user_all_rating[:] ** 2
                                cnt_node_t[user_all_items_rating_id[:], 0] += 1

                                sum_node_t_artist[user_all_items_rating_id_artist[:], 0] += user_all_rating_artist[:]
                                sum_square_node_t_artist[user_all_items_rating_id_artist[:], 0] += user_all_rating_artist[
                                                                                                 :] ** 2
                                cnt_node_t_artist[user_all_items_rating_id_artist[:], 0] += 1

                            # calculate sum_tL for node LIKE
                            elif valid_ratings_dislike:
                                sum_node_t[user_all_items_rating_id[:], 1] += user_all_rating[:]
                                sum_square_node_t[user_all_items_rating_id[:], 1] += user_all_rating[:] ** 2
                                cnt_node_t[user_all_items_rating_id[:], 1] += 1

                                sum_node_t_artist[user_all_items_rating_id_artist[:], 1] += user_all_rating_artist[:]
                                sum_square_node_t_artist[user_all_items_rating_id_artist[:], 1] += user_all_rating_artist[
                                                                                                 :] ** 2
                                cnt_node_t_artist[user_all_items_rating_id_artist[:], 1] += 1

                    # calculate sumtU for node UNKNOWN
                    sum_node_t[:, 2] = self.sum_cur_node_t[:] - sum_node_t[:, 0] - sum_node_t[:, 1]
                    sum_square_node_t[:, 2] = self.sum_cur_square_node_t[:] - sum_square_node_t[:, 0] - sum_square_node_t[:, 1]
                    cnt_node_t[:, 2] = self.cnt_cur_node_t[:] - cnt_node_t[:, 0] - cnt_node_t[:, 1]
                    sum_node_t_artist[:, 2] = self.sum_cur_node_t_artist[:] - sum_node_t_artist[:, 0] - sum_node_t_artist[:, 1]
                    sum_square_node_t_artist[:, 2] = self.sum_cur_square_node_t_artist[:] - sum_square_node_t_artist[:, 0] - sum_square_node_t_artist[:, 1]
                    cnt_node_t_artist[:, 2] = self.cnt_cur_node_t_artist[:] - cnt_node_t_artist[:, 0] - cnt_node_t_artist[:, 1]

                    error_total = self.calculate_item_error(sum_node_t, sum_square_node_t, cnt_node_t, sum_node_t_artist,
                                                            sum_square_node_t_artist, cnt_node_t_artist)
                    # print ("item_id: ", item_id, ", artist_id: ", artist_id, ", Error: ", error_total)
                    if error_total < min_error_item:
                        min_sum_node_t = sum_node_t
                        min_sum_square_node_t = sum_square_node_t
                        min_cnt_node_t = cnt_node_t
                        min_sum_node_t_artist = sum_node_t_artist
                        min_sum_square_node_t_artist = sum_square_node_t_artist
                        min_cnt_node_t_artist = cnt_node_t_artist
                        min_error_item = error_total
                        min_item_id = item_id
                        min_artist_id = artist_id

        # if not items_error_list:
        #    items_error_list[0] = 0
        return min_cnt_node_t, min_sum_node_t, min_sum_square_node_t, min_cnt_node_t_artist, min_sum_node_t_artist, min_sum_square_node_t_artist, min_item_id, min_artist_id

    def calculate_item_error(self, sum_node_t, sum_square_node_t, cnt_node_t, sum_node_t_artist, sum_square_node_t_artist,
                             cnt_node_t_artist):
        # Calculate error for one item-split in current node
        error_item = np.sum(sum_square_node_t - (sum_node_t ** 2) / (cnt_node_t + 1e-9))
        error_artist = np.sum(sum_square_node_t_artist - (sum_node_t_artist ** 2) / (cnt_node_t_artist + 1e-9))
        #consider equal weights for item error and artist error
        #avoid average for saving computation
        return error_item + error_artist

    def predict_fast(self, user_id, item_id_list):
        scores = {item: 0 for item in item_id_list}
        node = self.tree[:]
        current_node = node
        pred_index = 0
        for depth in range(len(self.split_items_list)):
            split_item = self.split_items_list[depth][pred_index]
            next_node_depth = str(depth + 1)
            if split_item in self.rU[user_id]:
                split_item_rating = self.rU[user_id][split_item]
                # print("split_item", split_item, ", rating", split_item_rating)
                if split_item_rating > 50:
                    pred_index = 3 * pred_index
                elif 50 >= split_item_rating >= 0:
                    pred_index = 3 * pred_index + 1
                else:
                    pred_index = 3 * pred_index + 2
            else:
                pred_index = 3 * pred_index + 2
            start_user_bounds = self.users_bound[next_node_depth][pred_index][0]
            end_user_bounds = self.users_bound[next_node_depth][pred_index][1]
            next_node = node[start_user_bounds: end_user_bounds + 1]
            #print("split_item, next_node_depth, pred_index,
            # start_user_bounds, end_user_bounds", split_item, next_node_depth, pred_index, start_user_bounds,  end_user_bounds)
            current_node = next_node
            if (start_user_bounds >= end_user_bounds):  # leaf_node
                break  # leaf node reached
        sums = defaultdict(int)
        counts = defaultdict(int)

        for u in current_node:
            for item_id, rating in self.rU[u].items():
                sums[item_id] += rating
                counts[item_id] += 1

        for item_id in item_id_list:
            if counts[item_id] != 0:
                scores[item_id] = sums[item_id] / counts[item_id]
        return scores

    def predict(self, user_id, item_id_list, min_ratings=5):
        scores = {item: 0 for item in item_id_list}
        node = self.tree[:]
        current_node = node
        node_depth = '0'
        pred_index = 0
        tree_index_list = {}
        tree_index_list[node_depth] = pred_index
        for depth in range(len(self.split_items_list)):
            split_item = self.split_items_list[depth][pred_index]
            node_depth = str(depth + 1)
            if split_item in self.rU[user_id]:
                split_item_rating = self.rU[user_id][split_item]
                if split_item_rating > 50:
                    pred_index = 3 * pred_index
                elif 50 >= split_item_rating >= 0:
                    pred_index = 3 * pred_index + 1
                else:
                    pred_index = 3 * pred_index + 2
            else:
                pred_index = 3 * pred_index + 2
            node_user_bounds = self.users_bound[node_depth][pred_index]
            current_node = node[node_user_bounds[0]: node_user_bounds[1] + 1]
            tree_index_list[node_depth] = pred_index
            if node_user_bounds[0] >= node_user_bounds[1]:  # leaf_node
                break  # leaf node reached

        sums = defaultdict(int)
        counts = defaultdict(int)

        for item_id in item_id_list:
            sums[item_id], counts[item_id] = self.traverse_node_for_min_ratings(
                tree_index_list, int(node_depth), current_node, item_id,
                min_ratings)

        for item_id in item_id_list:
            if counts[item_id] != 0:
                scores[item_id] = sums[item_id] / counts[item_id]
        return scores

    def traverse_node_for_min_ratings(self, tree_indexes, node_level,
                                      users_node, item_id, min_ratings):
        count_items = 0
        sum_items = 0
        # print(users_node)
        count_items = sum(
            1 for u in users_node if self.rU[u].get(item_id) is not None)
        sum_items = sum(self.rU[u][item_id] for u in users_node if
                        self.rU[u].get(item_id) is not None)

        if count_items < min_ratings:
            node_level = node_level - 1
            if node_level < 0:  # leaf_node
                return sum_items, count_items  # leaf node reached
            node_depth = str(node_level)
            pred_index = tree_indexes[node_depth]
            node_user_bounds = self.users_bound[node_depth][pred_index]
            parent_node = self.tree[
                          node_user_bounds[0]: node_user_bounds[1] + 1]
            sum_items, count_items = self.traverse_node_for_min_ratings(
                tree_indexes, node_level, parent_node, item_id, min_ratings)
        return sum_items, count_items

    def is_rating_count_pass(self, node_user_lst, item_id):
        num_rating = 0
        for u in node_user_lst:
            if item_id in self.rU[u]:
                num_rating += 1
        if num_rating > 5:
            return True
        else:
            return False

    def build_model(self):
        # Construct the tree & get the prediction model
        self.generate_decision_tree(self.users_bound['0'][0], [])
