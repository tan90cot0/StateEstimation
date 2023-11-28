import numpy as np
import math
from scipy.spatial.distance import euclidean, mahalanobis
from scipy.optimize import linear_sum_assignment
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import pandas as pd

def init_vars_q1():
    global A, B, C, threshold
    A = np.array([  [1, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 1],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1]])

    B = np.array([  [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])

    C = np.array([  [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0]])

    df = 3  # Example degrees of freedom
    alpha = 0.99  # Example significance level
    threshold = stats.chi2.ppf(alpha, df)

def calc_euclidean_dist(planes, observations):
    euclidean_distances = np.zeros((len(planes), len(observations)))
    for plane_index, plane in enumerate(planes):
        for obs_index, observation in enumerate(observations):
            euclidean_distances[plane_index, obs_index] = euclidean(plane.X[:3], observation)
    return euclidean_distances

def calc_mahalanobis_dist(planes, observations):
    mahalanobis_distances = np.zeros((len(planes), len(observations)))

    for plane_index, plane in enumerate(planes):
        for obs_index, observation in enumerate(observations):
            mahalanobis_distances[plane_index, obs_index] = mahalanobis(plane.X[:3], observation, np.linalg.inv(plane.sigma[:3, :3]))

    return mahalanobis_distances

def rearrange(planes, observations):
    return observations[linear_sum_assignment(calc_euclidean_dist(planes, observations))[1]]

def plot_3d(plane, fig = None, X_obs = None, X_est = None, X_actual = None, vx_actual = None, vx_est = None, filename = 'test'):
    if fig==None:
        fig = make_subplots()
    if X_obs is not None:
        df = pd.DataFrame({'x': plane.x_obs, 'y': plane.y_obs, 'z': plane.z_obs})
        fig.add_trace(go.Scatter3d(x=df['x'], y=df['y'], z=df['z'], name='Observed Position', mode = 'lines', line=dict(color=X_obs)))
    if X_est is not None:
        df = pd.DataFrame({'x': plane.x_est, 'y': plane.y_est, 'z': plane.z_est})
        fig.add_trace(go.Scatter3d(x=df['x'], y=df['y'], z=df['z'], name='Estimated Position', mode = 'lines', line=dict(color=X_est)))
    if X_actual is not None:
        df = pd.DataFrame({'x': plane.x_actual, 'y': plane.y_actual, 'z': plane.z_actual})
        fig.add_trace(go.Scatter3d(x=df['x'], y=df['y'], z=df['z'], name='Actual Position', mode = 'lines', line=dict(color=X_actual)))
    if vx_actual is not None:
        df = pd.DataFrame({'x': plane.ux_actual, 'y': plane.uy_actual, 'z': plane.uz_actual})
        fig.add_trace(go.Scatter3d(x=df['x'], y=df['y'], z=df['z'], name='Actual Speed', mode = 'lines', line=dict(color=vx_actual)))
    if vx_est is not None:
        df = pd.DataFrame({'x': plane.ux_belief, 'y': plane.uy_belief, 'z': plane.uz_belief})
        fig.add_trace(go.Scatter3d(x=df['x'], y=df['y'], z=df['z'], name='Estimated Speed', mode = 'lines', line=dict(color=vx_est)))
    if filename is not None:
        fig.write_html("plots/" + filename+ ".html")
    else:
        return fig

def plot_2d(plane, X_obs = None, X_est = None, X_actual = None, ellipse = None, t50 = None, t200 = None, t100 = None, filename = 'test'):
    fig = make_subplots()
    if ellipse is not None:
        fig.add_trace(go.Scatter(x=plane.ellipses_x, y=plane.ellipses_y, mode = 'lines', line=dict(color=ellipse), name='Uncertainty Ellipses'))
    if X_obs is not None:
        df = pd.DataFrame({'x': plane.x_obs, 'y': plane.y_obs})
        fig.add_trace(go.Scatter(x=df['x'], y=df['y'], name='Observed Position', mode = 'lines', line=dict(color=X_obs, width=1)))
    if X_est is not None:
        df = pd.DataFrame({'x': plane.x_est, 'y': plane.y_est})
        fig.add_trace(go.Scatter(x=df['x'], y=df['y'], name='Estimated Position', mode = 'lines', line=dict(color=X_est, width=1)))
    if X_actual is not None:
        df = pd.DataFrame({'x': plane.x_actual, 'y': plane.y_actual})
        fig.add_trace(go.Scatter(x=df['x'], y=df['y'], name='Actual Position', mode = 'lines', line=dict(color=X_actual, width=1)))
    if t50 is not None:
        df = pd.DataFrame({'x': plane.x_obs[49], 'y': [i for i in range(int(max(plane.y_obs))+1)]})
        fig.add_trace(go.Scatter(x=df['x'], y=df['y'], name='t = 50', mode = 'lines', line=dict(color=t50, width=1)))
        df = pd.DataFrame({'x': plane.x_obs[79], 'y': [i for i in range(int(max(plane.y_obs))+1)]})
        fig.add_trace(go.Scatter(x=df['x'], y=df['y'], name='t = 80', mode = 'lines', line=dict(color=t50, width=1)))
    if t200 is not None:
        df = pd.DataFrame({'x': plane.x_obs[199], 'y': [i for i in range(int(max(plane.y_obs))+1)]})
        fig.add_trace(go.Scatter(x=df['x'], y=df['y'], name='t = 200', mode = 'lines', line=dict(color=t200, width=1)))
        df = pd.DataFrame({'x': plane.x_obs[229], 'y': [i for i in range(int(max(plane.y_obs))+1)]})
        fig.add_trace(go.Scatter(x=df['x'], y=df['y'], name='t = 230', mode = 'lines', line=dict(color=t200, width=1)))
    if t100 is not None:
        df = pd.DataFrame({'x': plane.x_actual[99], 'y': [i for i in range(int(max(plane.y_actual))+1)]})
        fig.add_trace(go.Scatter(x=df['x'], y=df['y'], name='t = 100', mode = 'lines', line=dict(color=t100, width=1)))
    
    fig.write_html("plots/" + filename+ ".html")
    
class Plane:
    def __init__(self, x, y, z, vx, vy, vz, position_noise = 1, velocity_noise = 0.008, sensor_noise = 8):
        self.X = np.array([x, y, z, vx, vy, vz])
        
        #Prior belief
        self.mu = np.array([x, y, z, vx, vy, vz])
        self.sigma = 0.008 * np.identity(6)

        self.dims = 3
        self.ellipses_x = np.array([])
        self.ellipses_y = np.array([])

        self.x_actual = []
        self.y_actual = []
        self.z_actual = []
        self.x_obs = []
        self.y_obs = []
        self.z_obs = []
        self.x_est = []
        self.y_est = []
        self.z_est = []
        self.ux_actual = []
        self.uy_actual = []
        self.uz_actual = []
        self.ux_belief = []
        self.uy_belief = []
        self.uz_belief = []
    
        self.R = np.array([ [math.pow(position_noise, 2), 0, 0, 0, 0, 0],
                            [0, math.pow(position_noise, 2), 0, 0, 0, 0],
                            [0, 0, math.pow(position_noise, 2), 0, 0, 0],
                            [0, 0, 0, math.pow(velocity_noise, 2), 0, 0],
                            [0, 0, 0, 0, math.pow(velocity_noise, 2), 0],
                            [0, 0, 0, 0, 0, math.pow(velocity_noise, 2)]])
    
        self.Q = np.array([ [math.pow(sensor_noise, 2), 0, 0],
                            [0, math.pow(sensor_noise, 2), 0],
                            [0, 0, math.pow(sensor_noise, 2)]])

        # self.R = np.array([ [0.5 + np.random.random(), 0, 0, 0, 0, 0],
        #                     [0, 0.5 + np.random.random(), 0, 0, 0, 0],
        #                     [0, 0, 0.5 + np.random.random(), 0, 0, 0],
        #                     [0, 0, 0, 0.00005 + np.random.random() / 10000, 0, 0],
        #                     [0, 0, 0, 0, 0.00005 + np.random.random() / 10000, 0],
        #                     [0, 0, 0, 0, 0, 0.00005 + np.random.random() / 10000]])

        # self.Q = np.array([ [50 + np.random.random() * 100, 0, 0],
        #                     [0, 50 + np.random.random() * 100, 0],
        #                     [0, 0, 50 + np.random.random() * 100]])

    def update_graph_values(self, Z):
        self.x_actual.append(self.X[0])
        self.y_actual.append(self.X[1])
        self.z_actual.append(self.X[2])

        self.x_obs.append(Z[0])
        self.y_obs.append(Z[1])
        self.z_obs.append(Z[2])

        self.x_est.append(self.mu[0])
        self.y_est.append(self.mu[1])
        self.z_est.append(self.mu[2])

        self.ux_actual.append(self.X[3])
        self.uy_actual.append(self.X[4])
        self.uz_actual.append(self.X[5])

        self.ux_belief.append(self.mu[3])
        self.uy_belief.append(self.mu[4])
        self.uz_belief.append(self.mu[5])

    def get_state(self, ut):
        epsilon_t = np.random.multivariate_normal(np.zeros(self.dims*2), self.R)
        self.X = A @ self.X + B @ ut + epsilon_t

    def get_observation(self):
        delta_t = np.random.multivariate_normal(np.zeros(self.dims), self.Q)
        return C @ self.X + delta_t

    def kalman_filter(self, ut, Z, X_malfunction = False):
        mu = A @ self.mu + B @ ut
        sigma = A @ self.sigma @ np.transpose(A) + self.R
        K = sigma @ np.transpose(C) @ np.linalg.inv(C @ sigma @ np.transpose(C) + self.Q)
        self.mu = mu + K @ (Z - C @ mu)
        self.sigma = (np.identity(K.shape[0]) - K @ C) @ sigma
        if X_malfunction:
            self.mu[0] = mu[0]
            self.sigma[0] = sigma[0]

    def make_ellipses(self, N = 100):
        eigenvals, eigenvecs = np.linalg.eig(self.sigma[:2,:2])
        a, b = np.sqrt(eigenvals)
        t = np.linspace(0, 2*np.pi, N)
        xp, yp = eigenvecs.T @ [a * np.cos(t), b * np.sin(t)]
        self.ellipses_x = np.concatenate((self.ellipses_x, xp + self.mu[0]))
        self.ellipses_y = np.concatenate((self.ellipses_y, yp + self.mu[1]))

def ques_1a():
    plane = Plane(0, 0, 0, 1, 1, 1)
    for _ in range(300):
        plane.get_state(np.zeros(3))
        Z = plane.get_observation()
        plane.update_graph_values(Z)

    plot_3d(plane, X_obs='orange', X_actual='blue', filename='ques_1a')

def ques_1c():
    plane = Plane(0, 0, 0, 0, 0, 0)
    for t in range(300):
        ut = np.array([math.cos(t), math.sin(t), math.sin(t)])
        plane.get_state(ut)
        Z = plane.get_observation()
        plane.kalman_filter(ut, Z)
        plane.update_graph_values(Z)
        plane.make_ellipses()

    plot_2d(plane, X_obs='green', X_actual='blue', X_est='red', ellipse = 'black', filename='ques_1c_2d')
    plot_3d(plane, X_obs='green', X_actual='blue', X_est = 'red', filename='ques_1c_3d')

def ques_1d():
    plane = Plane(0, 0, 0, 0, 0, 0)
    filenames = ['ques_1d_position_increase', 'ques_1d_position_decrease', 'ques_1d_velocity_increase', 'ques_1d_velocity_decrease', 'ques_1d_sensor_increase', 'ques_1d_sensor_decrease']
    position_noises = [10, 0.1, 1, 1, 1, 1]
    velocity_noises = [0.008, 0.008, 0.08, 0.0008, 0.008, 0.008]
    sensor_noises = [8, 8, 8, 8, 80, 0.8]
    for ind, filename in enumerate(filenames):
        plane = Plane(0, 0, 0, 0, 0, 0, position_noises[ind], velocity_noises[ind], sensor_noises[ind])
        for t in range(300):
            ut = np.array([math.cos(t), math.sin(t), math.sin(t)])
            plane.get_state(ut)
            Z = plane.get_observation()
            plane.kalman_filter(ut, Z)
            plane.update_graph_values(Z)
            plane.make_ellipses()
        plot_2d(plane, X_obs='green', X_actual='blue', X_est='red', ellipse = 'black', filename=filename)
    
def ques_1e():
    plane = Plane(0, 0, 0, 0, 0, 0)
    for t in range(300):
        ut = np.array([math.cos(t), math.sin(t), math.sin(t)])
        plane.get_state(ut)
        if not((t>=50 and t<80) or (t>=200 and t<230)):
            Z = plane.get_observation()
        plane.kalman_filter(ut, Z)
        plane.update_graph_values(Z)
        plane.make_ellipses()
    
    plot_2d(plane, X_obs='green', X_actual='blue', X_est='red', ellipse = 'black', t50 ='purple', t200 = 'orange', filename='ques_1e')

def ques_1f():
    plane = Plane(0, 0, 0, 0, 0, 0)
    for t in range(300):
        ut = np.array([math.cos(t), math.sin(t), math.sin(t)])
        plane.get_state(ut)
        Z = plane.get_observation()
        if t>=100:
            plane.kalman_filter(ut, Z, True)
        else:
            plane.kalman_filter(ut, Z)
        plane.update_graph_values(Z)
        plane.make_ellipses()
    
    plot_2d(plane, X_actual='blue', X_est='red', ellipse = 'black', t100 ='purple', filename='ques_1f')

def ques_1g():
    plane = Plane(0, 0, 0, 0, 0, 0)
    for t in range(50):
        ut = np.array([math.cos(t), math.sin(t), math.sin(t)])
        plane.get_state(ut)
        Z = plane.get_observation()
        plane.kalman_filter(ut, Z)
        plane.update_graph_values(Z)

    plot_3d(plane, vx_actual='blue',vx_est = 'red', filename='ques_1g')

def ques_1h():
    planes = []
    num_planes = 2
    for _ in range(num_planes):
        planes.append(Plane(np.random.randint(0, 10), np.random.randint(0, 10), np.random.randint(0, 10), 0, 0, 0))
    
    for t in range(300):
        ut = np.array([math.cos(t), math.sin(t), math.sin(t)])
        observations = np.zeros((num_planes, 3))
        for ind, plane in enumerate(planes):
            plane.get_state(ut)
            observations[ind] = plane.get_observation()
            
        np.random.shuffle(observations)
        # Nearest Neighbour
        # observations = rearrange(planes, observations)
        # for ind, plane in enumerate(planes):
        #     plane.kalman_filter(ut, observations[ind])
        #     plane.update_graph_values(observations[ind])

        # Mahalanobis Distance
        mahalanobis_distances = calc_mahalanobis_dist(planes, observations)
        _, col_ind = linear_sum_assignment(mahalanobis_distances)

        for ind, plane in enumerate(planes):
            observation_index = col_ind[ind]
            if mahalanobis_distances[ind, observation_index] < 36:
                plane.kalman_filter(ut, observations[observation_index])
                plane.update_graph_values(observations[observation_index])
    
    fig = None
    for plane in planes:
        fig = plot_3d(plane = plane, fig = fig, X_actual='blue', X_est = 'red', filename=None)
    fig.write_html("plots/ques_1h.html")

def ques_1i():
    planes = []
    num_planes = 5
    for _ in range(num_planes):
        planes.append(Plane(np.random.randint(0, 10), np.random.randint(0, 10), np.random.randint(0, 10), 0, 0, 0))
    
    for t in range(300):
        ut = np.array([math.cos(t), math.sin(t), math.sin(t)])
        observations = np.zeros((num_planes, 3))
        for ind, plane in enumerate(planes):
            plane.get_state(ut)
            observations[ind] = plane.get_observation()
            
        np.random.shuffle(observations)
        # Nearest Neighbour
        # observations = rearrange(planes, observations)
        # for ind, plane in enumerate(planes):
        #     plane.kalman_filter(ut, observations[ind])
        #     plane.update_graph_values(observations[ind])

        # Mahalanobis Distance
        mahalanobis_distances = calc_mahalanobis_dist(planes, observations)
        _, col_ind = linear_sum_assignment(mahalanobis_distances)

        for ind, plane in enumerate(planes):
            observation_index = col_ind[ind]
            if mahalanobis_distances[ind, observation_index] < 36:
                plane.kalman_filter(ut, observations[observation_index])
                plane.update_graph_values(observations[observation_index])
    
    fig = None
    for plane in planes:
        fig = plot_3d(plane = plane, fig = fig, X_actual='blue', X_est = 'red', filename=None)
    fig.write_html("plots/ques_1i.html")


def init_vars_q2():
    #Frequency = 1Hz
    global dt, A, B, C
    dt = 1
    A = np.array([  [1, 0, dt, 0],
                    [0, 1, 0, dt],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    B = np.array([  [0, 0],
                    [0, 0],
                    [1, 0],
                    [0, 1],])

    C = np.array([  [1, 0, 0, 0],
                    [0, 1, 0, 0]])

def h(x, nearest_point): 
    return np.array([x[0], x[1], np.linalg.norm(x[:2] - nearest_point)])

class Plane2:
    def __init__(self, x, y, vx, vy, motion_error):
        self.X = np.array([x, y, vx, vy])
        self.x_actual = []
        self.y_actual = []
        self.x_est = []
        self.y_est = []
        self.x_obs = []
        self.y_obs = []
        self.ellipses_x = np.array([])
        self.ellipses_y = np.array([])
        self.mu = self.X.copy()
        self.sigma = np.zeros((4, 4))
        self.R = (motion_error ** 2) * np.identity(4)
        self.Q = None

    def update_state(self, u):
        self.X = A @ self.X + B @ u + np.random.multivariate_normal(np.zeros(4), self.R)

    def get_observation(self, nearest_point):
        if nearest_point is not None:
            return h(self.X, nearest_point) + np.random.multivariate_normal(np.zeros(3), self.Q)
        return C @ self.X + np.random.multivariate_normal(np.zeros(2), self.Q)

    def extended_kalman_filter_update(self, ut, Z, nearest_point):
        mu = A @ self.mu + B @ ut
        sigma = A @ self.sigma @ np.transpose(A) + self.R

        if nearest_point is None:
            K = sigma @ np.transpose(C) @ np.linalg.inv(C @ sigma @ np.transpose(C) + self.Q)
            self.mu = mu + K @ (Z - C @ mu)
            self.sigma = (np.identity(K.shape[0]) - K @ C) @ sigma
        else:
            x, y = mu[:2]
            xi, yi = nearest_point
            norm = np.linalg.norm(mu[:2] - nearest_point)
            H =  np.array([ [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [(x-xi) / norm if norm > 0 else np.inf, (y-yi) / norm if norm > 0 else np.inf, 0, 0]])
            K = sigma @ np.transpose(H) @ np.linalg.inv(H @ sigma @ np.transpose(H) + self.Q)
            self.mu = mu + K @ (Z - h(mu[:2], nearest_point))
            self.sigma = (np.identity(4) - K @ H) @ sigma
    
    def get_nearest_landmark(self, landmarks, observation_range):
        distances = [(np.linalg.norm(self.X[:2] - landmarks[i]), i) for i in range(len(landmarks))]
        distances.sort()
        if distances[0][0] > observation_range:
            return None
        else:
            return landmarks[distances[0][1]]
    
    def make_ellipses(self, N = 100):
        eigenvals, eigenvecs = np.linalg.eig(self.sigma[:2,:2])
        a, b = np.sqrt(eigenvals)
        t = np.linspace(0, 2*np.pi, N)
        xp, yp = eigenvecs.T @ [a * np.cos(t), b * np.sin(t)]
        self.ellipses_x = np.concatenate((self.ellipses_x, xp + self.mu[0]))
        self.ellipses_y = np.concatenate((self.ellipses_y, yp + self.mu[1]))

    def update_params(self, Z):
        self.x_actual.append(self.X[0])
        self.y_actual.append(self.X[1])

        self.x_obs.append(Z[0])
        self.y_obs.append(Z[1])

        self.x_est.append(self.mu[0])
        self.y_est.append(self.mu[1])

    def plot(self, landmarks):
        landmarks_x = [p[0] for p in landmarks]
        landmarks_y = [p[1] for p in landmarks]
        fig = make_subplots()

        df = pd.DataFrame({'x': self.x_actual, 'y': self.y_actual})
        fig.add_trace(go.Scatter(x=df['x'], y=df['y'], name='Actual Position', mode = 'lines', line=dict(color='blue', width=1)))

        df = pd.DataFrame({'x': self.x_est, 'y': self.y_est})
        fig.add_trace(go.Scatter(x=df['x'], y=df['y'], name='Estimated Position', mode = 'lines', line=dict(color='green', width=1)))

        df = pd.DataFrame({'x': self.x_obs, 'y': self.y_obs})
        fig.add_trace(go.Scatter(x=df['x'], y=df['y'], name='Observed Position', mode = 'lines', line=dict(color='red', width=1)))

        df = pd.DataFrame({'x': landmarks_x, 'y': landmarks_y})
        fig.add_trace(go.Scatter(x=df['x'], y=df['y'], name='Landmarks', mode = 'markers'))

        fig.add_trace(go.Scatter(x=self.ellipses_x, y=self.ellipses_y, mode = 'lines', line=dict(color='purple'), name='Uncertainty Ellipses'))

        def plot_circle(x, y, radius, color, ind):
            theta = np.linspace(0, 2 * np.pi, 100)
            x_circle = x + radius * np.cos(theta)
            y_circle = y + radius * np.sin(theta)
            fig.add_trace(go.Scatter(x=x_circle, y=y_circle, name = "Landmark {}".format(ind), mode='lines', line=dict(color=color, width=1)))
        
        for ind, point in enumerate(landmarks):
            plot_circle(point[0], point[1], 50, "black", ind)

        fig.update_layout(xaxis=dict(range=[-500, 4500]), yaxis=dict(range=[-500, 3000]))
        fig.write_html("plots/ques_2.html")

def landmark_localisation(timesteps, initial_position, speed, heading_direction, landmark_error, gps_error, motion_error, observation_range, landmarks):
    plane = Plane2(initial_position[0], initial_position[1], speed * math.cos(heading_direction), speed * math.sin(heading_direction), motion_error)
    for t in range(0, timesteps, dt):
        ut = np.array([math.cos(t), math.sin(t)])
        plane.update_state(ut)
        nearest_point = plane.get_nearest_landmark(landmarks, observation_range)
        if nearest_point is not None:
            plane.Q = np.array([[gps_error**2, 0, 0],
                                [0, gps_error**2, 0],
                                [0, 0, landmark_error ** 2]])
        else:
            plane.Q = np.array([[gps_error**2, 0],
                                [0, gps_error**2]])
        z_t = plane.get_observation(nearest_point)
        plane.extended_kalman_filter_update(ut, z_t, nearest_point)
        plane.make_ellipses()
        plane.update_params(z_t)
    plane.plot(landmarks)

def ques_2():
    init_vars_q2()
    landmark_localisation(timesteps=1000, 
                      initial_position=np.array([-200, -50]), 
                      speed = 4, 
                      heading_direction = 0.35, 
                      landmark_error=200, 
                      gps_error=10, 
                      motion_error=0.01, 
                      observation_range=50, 
                      landmarks=[np.array([150, 0]),np.array([-150, 0]),np.array([0, 150]),np.array([0, -150]),np.array([25, 0])])
    
init_vars_q1()
ques_1a()
ques_1c()
ques_1d()
ques_1e()
ques_1f()
ques_1g()
ques_1h()
ques_1i()
ques_2()