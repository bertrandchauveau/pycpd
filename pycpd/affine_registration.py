from builtins import super
import cupy as cp
from .emregistration import EMRegistration
from .utility import is_positive_semi_definite


class AffineRegistration(EMRegistration):
    """
    Affine registration.

    Attributes
    ----------
    B: numpy array (semi-positive definite)
        DxD affine transformation matrix.

    t: numpy array
        1xD initial translation vector.
    """
    # Additional parameters used in this class, but not inputs.
    # YPY: float
    #     Denominator value used to update the scale factor.
    #     Defined in Fig. 2 and Eq. 8 of https://arxiv.org/pdf/0905.2635.pdf.

    # X_hat: numpy array
    #     Centered target point cloud.
    #     Defined in Fig. 2 of https://arxiv.org/pdf/0905.2635.pdf

    def __init__(self, B=None, t=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if B is not None and ((B.ndim != 2) or (B.shape[0] != self.D) or (B.shape[1] != self.D) or not is_positive_semi_definite(B)):
            raise ValueError(
                'The rotation matrix can only be initialized to {}x{} positive semi definite matrices. Instead got: {}.'.format(self.D, self.D, B))

        if t is not None and ((t.ndim != 2) or (t.shape[0] != 1) or (t.shape[1] != self.D)):
            raise ValueError(
                'The translation vector can only be initialized to 1x{} positive semi definite matrices. Instead got: {}.'.format(self.D, t))
        
        self.B = cp.eye(self.D) if B is None else B
        self.t = cp.atleast_2d(cp.zeros((1, self.D))) if t is None else t

        self.YPY = None
        self.X_hat = None
        self.A = None

    def update_transform(self):
        """
        Calculate a new estimate of the rigid transformation.

        """

        # source and target point cloud means
        muX = cp.divide(cp.sum(self.PX, axis=0), self.Np)
        muY = cp.divide(
            cp.sum(cp.dot(cp.transpose(self.P), self.Y), axis=0), self.Np)

        self.X_hat = self.X - cp.tile(muX, (self.N, 1))
        Y_hat = self.Y - cp.tile(muY, (self.M, 1))

        self.A = cp.dot(cp.transpose(self.X_hat), cp.transpose(self.P))
        self.A = cp.dot(self.A, Y_hat)

        self.YPY = cp.dot(cp.transpose(Y_hat), cp.diag(self.P1))
        self.YPY = cp.dot(self.YPY, Y_hat)

        # Calculate the new estimate of affine parameters using update rules for (B, t)
        # as defined in Fig. 3 of https://arxiv.org/pdf/0905.2635.pdf.
        self.B = cp.linalg.solve(cp.transpose(self.YPY), cp.transpose(self.A))
        self.t = cp.transpose(
            muX) - cp.dot(cp.transpose(self.B), cp.transpose(muY))

    def transform_point_cloud(self, Y=None):
        """
        Update a point cloud using the new estimate of the affine transformation.
        
        Attributes
        ----------
        Y: numpy array, optional
            Array of points to transform - use to predict on new set of points.
            Best for predicting on new points not used to run initial registration.
                If None, self.Y used.
        
        Returns
        -------
        If Y is None, returns None.
        Otherwise, returns the transformed Y.

        """
        if Y is None:
            self.TY = cp.dot(self.Y, self.B) + cp.tile(self.t, (self.M, 1))
            return
        else:
            return cp.dot(Y, self.B) + cp.tile(self.t, (Y.shape[0], 1))

    def update_variance(self):
        """
        Update the variance of the mixture model using the new estimate of the affine transformation.
        See the update rule for sigma2 in Fig. 3 of of https://arxiv.org/pdf/0905.2635.pdf.

        """
        qprev = self.q

        trAB = cp.trace(cp.dot(self.A, self.B))
        xPx = cp.dot(cp.transpose(self.Pt1), cp.sum(
            cp.multiply(self.X_hat, self.X_hat), axis=1))
        trBYPYP = cp.trace(cp.dot(cp.dot(self.B, self.YPY), self.B))
        self.q = (xPx - 2 * trAB + trBYPYP) / (2 * self.sigma2) + \
            self.D * self.Np/2 * cp.log(self.sigma2)
        self.diff = cp.abs(self.q - qprev)

        self.sigma2 = (xPx - trAB) / (self.Np * self.D)

        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

    def get_registration_parameters(self):
        """
        Return the current estimate of the affine transformation parameters.

        Returns
        -------
        B: numpy array
            DxD affine transformation matrix.

        t: numpy array
            1xD translation vector.

        """
        return self.B, self.t
