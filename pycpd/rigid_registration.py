from builtins import super
import cupy as cp
import numbers
from .emregistration import EMRegistration
from .utility import is_positive_semi_definite


class RigidRegistration(EMRegistration):
    """
    Rigid registration.

    Attributes
    ----------
    R: numpy array (semi-positive definite)
        DxD rotation matrix. Any well behaved matrix will do,
        since the next estimate is a rotation matrix.

    t: numpy array
        1xD initial translation vector.

    s: float (positive)
        scaling parameter.

    A: numpy array
        Utility array used to calculate the rotation matrix.
        Defined in Fig. 2 of https://arxiv.org/pdf/0905.2635.pdf.

    """
    # Additional parameters used in this class, but not inputs.
    # YPY: float
    #     Denominator value used to update the scale factor.
    #     Defined in Fig. 2 and Eq. 8 of https://arxiv.org/pdf/0905.2635.pdf.

    # X_hat: numpy array
    #     Centered target point cloud.
    #     Defined in Fig. 2 of https://arxiv.org/pdf/0905.2635.pdf.


    def __init__(self, R=None, t=None, s=None, scale=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.D != 2 and self.D != 3:
            raise ValueError(
                'Rigid registration only supports 2D or 3D point clouds. Instead got {}.'.format(self.D))

        if R is not None and ((R.ndim != 2) or (R.shape[0] != self.D) or (R.shape[1] != self.D) or not is_positive_semi_definite(R)):
            raise ValueError(
                'The rotation matrix can only be initialized to {}x{} positive semi definite matrices. Instead got: {}.'.format(self.D, self.D, R))

        if t is not None and ((t.ndim != 2) or (t.shape[0] != 1) or (t.shape[1] != self.D)):
            raise ValueError(
                'The translation vector can only be initialized to 1x{} positive semi definite matrices. Instead got: {}.'.format(self.D, t))

        if s is not None and (not isinstance(s, numbers.Number) or s <= 0):
            raise ValueError(
                'The scale factor must be a positive number. Instead got: {}.'.format(s))

        self.R = cp.eye(self.D) if R is None else R
        self.t = cp.atleast_2d(cp.zeros((1, self.D))) if t is None else t
        self.s = 1 if s is None else s
        self.scale = scale

    def update_transform(self):
        """
        Calculate a new estimate of the rigid transformation.

        """

        # target point cloud mean
        muX = cp.divide(cp.sum(self.PX, axis=0),
                        self.Np)
        # source point cloud mean
        muY = cp.divide(
            cp.sum(cp.dot(cp.transpose(self.P), self.Y), axis=0), self.Np)

        self.X_hat = self.X - cp.tile(muX, (self.N, 1))
        # centered source point cloud
        Y_hat = self.Y - cp.tile(muY, (self.M, 1))
        self.YPY = cp.dot(cp.transpose(self.P1), cp.sum(
            cp.multiply(Y_hat, Y_hat), axis=1))

        self.A = cp.dot(cp.transpose(self.X_hat), cp.transpose(self.P))
        self.A = cp.dot(self.A, Y_hat)

        # Singular value decomposition as per lemma 1 of https://arxiv.org/pdf/0905.2635.pdf.
        U, _, V = cp.linalg.svd(self.A, full_matrices=True)
        C = cp.ones((self.D, ))
        C[self.D-1] = cp.linalg.det(cp.dot(U, V))

        # Calculate the rotation matrix using Eq. 9 of https://arxiv.org/pdf/0905.2635.pdf.
        self.R = cp.transpose(cp.dot(cp.dot(U, cp.diag(C)), V))
        # Update scale and translation using Fig. 2 of https://arxiv.org/pdf/0905.2635.pdf.
        if self.scale is True:
            self.s = cp.trace(cp.dot(cp.transpose(self.A), cp.transpose(self.R))) / self.YPY
        else:
            pass
        self.t = cp.transpose(muX) - self.s * \
            cp.dot(cp.transpose(self.R), cp.transpose(muY))

    def transform_point_cloud(self, Y=None):
        """
        Update a point cloud using the new estimate of the rigid transformation.

        Attributes
        ----------
        Y: numpy array
            Point cloud to be transformed - use to predict on new set of points.
            Best for predicting on new points not used to run initial registration.
                If None, self.Y used.
        
        
        Returns
        -------
        If Y is None, returns None.
        Otherwise, returns the transformed Y.
        """
        if Y is None:
            self.TY = self.s * cp.dot(self.Y, self.R) + self.t
            return
        else:
            return self.s * cp.dot(Y, self.R) + self.t

    def update_variance(self):
        """
        Update the variance of the mixture model using the new estimate of the rigid transformation.
        See the update rule for sigma2 in Fig. 2 of of https://arxiv.org/pdf/0905.2635.pdf.

        """
        qprev = self.q

        trAR = cp.trace(cp.dot(self.A, self.R))
        xPx = cp.dot(cp.transpose(self.Pt1), cp.sum(
            cp.multiply(self.X_hat, self.X_hat), axis=1))
        self.q = (xPx - 2 * self.s * trAR + self.s * self.s * self.YPY) / \
            (2 * self.sigma2) + self.D * self.Np/2 * cp.log(self.sigma2)
        self.diff = cp.abs(self.q - qprev)
        self.sigma2 = (xPx - self.s * trAR) / (self.Np * self.D)
        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

    def get_registration_parameters(self):
        """
        Return the current estimate of the rigid transformation parameters.

        Returns
        -------
        self.s: float
            Current estimate of the scale factor.
        
        self.R: numpy array
            Current estimate of the rotation matrix.
        
        self.t: numpy array
            Current estimate of the translation vector.
        """
        return self.s, self.R, self.t
