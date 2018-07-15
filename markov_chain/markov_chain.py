import numpy as np
from scipy.sparse.linalg import eigs

# champion prediction using markov chain
# score file store the score of every matches (.csv)
# team file store the team name with id(same with score file) (.txt)
# 1st step: init boject using two files
# 2nd step: train()
# result stored in .res_score
class markov_chain(object):
    
    def __init__(self, score_file, team_file):
        
        # get team name
        with open(team_file, 'r') as f:
            self.team_name = [line.strip() for line in f]
        self.team_name = np.array(self.team_name)
        self.team_num = self.team_name.shape[0]
        
        # init M
        self.generate_mat(score_file)

        # calcualte w_inf
        self.get_w_inf()

    def generate_mat(self, file):

        self.M = np.zeros((self.team_num, self.team_num))

        with open(file) as f:
            for row in f:
                team_a, pt_a, team_b, pt_b = [int(x) for x in row.rstrip('\n').split(',')]

                a_weight = pt_a/(pt_a+pt_b)
                a_wins = int(pt_a > pt_b)
                i = team_a - 1
                j = team_b - 1

                self.M[i,i] += a_wins + a_weight
                self.M[j,i] += a_wins + a_weight
                self.M[j,j] += 2 - a_wins - a_weight
                self.M[i,j] += 2 - a_wins - a_weight

        self.M = self.M / np.sum(self.M, axis=1).reshape(-1,1)

    def get_w_inf(self):
        self.w_inf = eigs(self.M.T,1)[1].flatten()
        self.w_inf = self.w_inf / np.sum(self.w_inf)


    def train(self, itr=10000, record=10):
        r = record
        w = np.repeat(1/self.team_num, self.team_num)
        self.res_team = []
        # final result, team with highest score is predicted to be champion
        self.res_score = []
        self.res_delta = []
        for i in range(itr):
            w = np.dot(w, self.M)
            self.res_delta.append(np.sum(np.abs(w-self.w_inf)))
            if(i==r-1):
                print(i)
                r *= record
                team_rank = np.argsort(w)[::-1][:25]
                team_name = [str(self.team_name[n]) for n in team_rank]
                team_score = [w[n] for n in team_rank]
                self.res_team.append(team_name)
                self.res_score.append(team_score)