import numpy as np

def standardise_matrix(original):
	"""Task 1: matrix standardisation."""
	means = np.mean(original, axis=0)
	stds  = np.std(original, axis=0)

	return (original - means) / stds

def pairwise_distances(P, Q):
	"""Task 2: pairwise distances in the plane"""
	diff_dim0 = np.tile(P[:,0], (Q.shape[0], 1)).T - Q[:,0]
	diff_dim1 = np.tile(P[:,1], (Q.shape[0], 1)).T - Q[:,1]
	D = np.sqrt(np.square(diff_dim0) + np.square(diff_dim1))

	return D

def sample_likelihood(X, theta1, theta2):
	"""Task 3: likelihood of a data sample"""
	mu1, sigma1 = theta1
	mu2, sigma2 = theta2

	d = X.shape[1]
	pi = np.pi

	X_minus_mu1 = X - mu1
	X_minus_mu2 = X - mu2

	first_likelihood = 1.0 / (np.sqrt(np.linalg.det(sigma1))) * np.exp(-0.5 * np.diagonal(X_minus_mu1.dot(np.linalg.inv(sigma1)).dot(X_minus_mu1.T)))
	second_likelihood = 1.0 / (np.sqrt(np.linalg.det(sigma2))) * np.exp(-0.5 * np.diagonal(X_minus_mu2.dot(np.linalg.inv(sigma2)).dot(X_minus_mu2.T)))

	first_likelier = first_likelihood - second_likelihood
	return list(map(lambda x: 1 if x>0 else 2, first_likelier))

	

if __name__ == "__main__":

	A = np.asarray(
		[[1, 2, 0],
		 [0, 25, 0],
		 [-1, 0, 0.1]
		])

	print(standardise_matrix(A))

	P = np.asarray(
		[[0, 0],
		 [1, 2],
		 [3, 3]
		])
	Q = np.asarray(
		[[0, 0],
		 [1, 1]
		])

	print(pairwise_distances(P, Q))

	X = np.asarray(
		[[0, 0],
		 [1, 2],
		 [3, 3]
		])
	theta1 = (0, np.asarray([[1, 0], [0, 1]]))
	theta2 = (3, np.asarray([[1, 0], [0, 1]]))

	print(sample_likelihood(X, theta1, theta2))
