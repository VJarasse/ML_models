import numpy as np


class cluster():
    """
    Class representing a cluster in the k-means algorithm
    """

    def __init__(self, coordinates):
        """
        Initialisation of the cluster : assignation of its center and initialisation of its members
        :param coordinates: tuple of the coordinates (for all dimensions) of the cluster center
        """
        self.center = coordinates
        # Members list is originally empty. It is a matrix a size [nb members x nb dimensions]
        self.members = None
        self.ids = None
        self.size = 0

    def add_member(self, id, coordinates):
        """
        Add one new point to a cluster
        :param coordinates: coordinates of the point to add
        :return: True if execution was a success, False if not.
        """
        if self.size == 0:
            self.ids = np.array([id])
            self.members = np.array(coordinates).reshape((1, len(coordinates)))
            self.size += 1
            return True
        else:
            if coordinates not in self.members:
                self.members = np.append(self.members, np.array(coordinates).reshape((1, len(coordinates))),
                                         axis=0)
                self.ids = np.append(self.ids, id)
                self.size += 1
                return True
            else:
                return False

    def remove_member(self, id):
        """
        Remove a point from a cluster
        :param coordinates: coordinates of the point to be removed
        :return: True if execution was a success, False if not.
        """
        if self.size < 1:
            return False
        else:
            if id in self.ids:
                i = int(np.where(self.ids == id)[0])
                self.ids = np.delete(self.ids, i)
                self.members = np.delete(self.members, i, axis=0)
                self.size -= 1
                return True
            else:
                return False

    def get_mean(self):
        """
        Compute the mean coordinates of all members of the cluster
        :return: coordinates of the mean of the cluster
        """
        return np.mean(self.members, axis=0)

    def update_center(self):
        """
        Uses get_mean to update the center of the cluster
        :return: nothing
        """
        self.center = self.get_mean()
