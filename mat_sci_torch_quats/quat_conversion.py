####################################################################################################
# Copyright (c) 2017-2020, Martin Diehl/Max-Planck-Institut für Eisenforschung GmbH
# Copyright (c) 2013-2014, Marc De Graef/Carnegie Mellon University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the following conditions are met:
#
#     - Redistributions of source code must retain the above copyright notice, this list
#        of conditions and the following disclaimer.
#     - Redistributions in binary form must reproduce the above copyright notice, this
#        list of conditions and the following disclaimer in the documentation and/or
#        other materials provided with the distribution.
#     - Neither the names of Marc De Graef, Carnegie Mellon University nor the names
#        of its contributors may be used to endorse or promote products derived from
#        this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
####################################################################################################

# Note: An object oriented approach to use this conversions is available in DAMASK, see
# https://damask.mpie.de and https://github.com/eisenforschung/DAMASK


import numpy as np

P = -1

# parameters for conversion from/to cubochoric
sc   = np.pi**(1./6.)/6.**(1./6.)
beta = np.pi**(5./6.)/6.**(1./6.)/2.
R1   = (3.*np.pi/4.)**(1./3.)


#---------- Quaternion Operations ----------

def quat2cubo(qu, scalar_first=True ):
    """Quaternion to cubochoric vector."""

    """ Step 1: Quaternion to homochoric vector."""

    # Converting Quaternion Order convention
    # DREAM3D convention is <xyz> s (<xyz> = imaginary vector component)
    # Convention used here is s <xyz>
    # This conversion is <xyz>,s to s,<xyz>
    if scalar_first is not True:
        tempqu = np.transpose(np.asarray([qu[:, 3], qu[:, 0], qu[:, 1], qu[:, 2]]))
        qu = tempqu
    #-------

    with np.errstate(invalid='ignore'):
        omega = 2.0 * np.arccos(np.clip(qu[..., 0:1], -1.0, 1.0))
        ho = np.where(np.abs(omega) < 1.0e-12,
                      np.zeros(3),
                      qu[..., 1:4] / np.linalg.norm(qu[..., 1:4], axis=-1, keepdims=True) \
                      * np.cbrt(0.75 * (omega - np.sin(omega))))

    # return ho # inserted here gives back the homochoric coordinates

    """
        Step 2: Homochoric vector to cubochoric vector.

        References
        ----------
        D. Roşca et al., Modelling and Simulation in Materials Science and Engineering 22:075013, 2014
        https://doi.org/10.1088/0965-0393/22/7/075013

        """
    rs = np.linalg.norm(ho, axis=-1, keepdims=True)

    xyz3 = np.take_along_axis(ho, _get_pyramid_order(ho, 'forward'), -1)

    with np.errstate(invalid='ignore', divide='ignore'):
        # inverse M_3
        xyz2 = xyz3[..., 0:2] * np.sqrt(2.0 * rs / (rs + np.abs(xyz3[..., 2:3])))
        qxy = np.sum(xyz2 ** 2, axis=-1, keepdims=True)

        q2 = qxy + np.max(np.abs(xyz2), axis=-1, keepdims=True) ** 2
        sq2 = np.sqrt(q2)
        q = (beta / np.sqrt(2.0) / R1) * np.sqrt(q2 * qxy / (q2 - np.max(np.abs(xyz2), axis=-1, keepdims=True) * sq2))
        tt = np.clip((np.min(np.abs(xyz2), axis=-1, keepdims=True) ** 2 \
                      + np.max(np.abs(xyz2), axis=-1, keepdims=True) * sq2) / np.sqrt(2.0) / qxy, -1.0, 1.0)
        T_inv = np.where(np.abs(xyz2[..., 1:2]) <= np.abs(xyz2[..., 0:1]),
                         np.block([np.ones_like(tt), np.arccos(tt) / np.pi * 12.0]),
                         np.block([np.arccos(tt) / np.pi * 12.0, np.ones_like(tt)])) * q
        T_inv[xyz2 < 0.0] *= -1.0
        T_inv[np.broadcast_to(np.isclose(qxy, 0.0, rtol=0.0, atol=1.0e-12), T_inv.shape)] = 0.0
        cu = np.block(
            [T_inv, np.where(xyz3[..., 2:3] < 0.0, -np.ones_like(xyz3[..., 2:3]), np.ones_like(xyz3[..., 2:3])) \
             * rs / np.sqrt(6.0 / np.pi),
             ]) / sc

    cu[np.isclose(np.sum(np.abs(ho), axis=-1), 0.0, rtol=0.0, atol=1.0e-16)] = 0.0
    cu = np.take_along_axis(cu, _get_pyramid_order(ho, 'backward'), -1)

    return cu

def quat2rod(qu, scalar_first=True):
    """Step 1: Quaternion to Rodrigues-Frank vector."""

    # Converting Quaternion Order convention
    # DREAM3D convention is <xyz> s (<xyz> = imaginary vector component)
    # Convention used here is s <xyz>
    # This conversion is <xyz>,s to s,<xyz>
    if scalar_first is not True:
        tempqu = np.transpose(np.asarray([qu[:, 3], qu[:, 0], qu[:, 1], qu[:, 2]]))
        qu = tempqu
    # -------

    with np.errstate(invalid='ignore', divide='ignore'):
        s = np.linalg.norm(qu[..., 1:4], axis=-1, keepdims=True)
        ro = np.where(np.broadcast_to(np.abs(qu[..., 0:1]) < 1.0e-12, qu.shape),
                      np.block([qu[..., 1:2], qu[..., 2:3], qu[..., 3:4],
                                np.broadcast_to(np.inf, qu.shape[:-1] + (1,))]),
                      np.block([qu[..., 1:2] / s, qu[..., 2:3] / s, qu[..., 3:4] / s,
                                np.tan(np.arccos(np.clip(qu[..., 0:1], -1.0, 1.0)))
                                ])
                      )
    ro[np.abs(s).squeeze(-1) < 1.0e-12] = [0.0, 0.0, P, 0.0]
    return ro

#---------- Cubochoric Operations ----------

def cubo2quat(cu, scalar_first=True):
    """Cubochoric vector to quaternion."""

    """
        Step 1: Cubochoric vector to homochoric vector.

        References
        ----------
        D. Roşca et al., Modelling and Simulation in Materials Science and Engineering 22:075013, 2014
        https://doi.org/10.1088/0965-0393/22/7/075013

        """
    with np.errstate(invalid='ignore', divide='ignore'):
        # get pyramide and scale by grid parameter ratio
        XYZ = np.take_along_axis(cu, _get_pyramid_order(cu, 'forward'), -1) * sc
        order = np.abs(XYZ[..., 1:2]) <= np.abs(XYZ[..., 0:1])
        q = np.pi / 12.0 * np.where(order, XYZ[..., 1:2], XYZ[..., 0:1]) \
            / np.where(order, XYZ[..., 0:1], XYZ[..., 1:2])
        c = np.cos(q)
        s = np.sin(q)
        q = R1 * 2.0 ** 0.25 / beta / np.sqrt(np.sqrt(2.0) - c) \
            * np.where(order, XYZ[..., 0:1], XYZ[..., 1:2])

        T = np.block([(np.sqrt(2.0) * c - 1.0), np.sqrt(2.0) * s]) * q

        # transform to sphere grid (inverse Lambert)
        c = np.sum(T ** 2, axis=-1, keepdims=True)
        s = c * np.pi / 24.0 / XYZ[..., 2:3] ** 2
        c = c * np.sqrt(np.pi / 24.0) / XYZ[..., 2:3]
        q = np.sqrt(1.0 - s)

        ho = np.where(np.isclose(np.sum(np.abs(XYZ[..., 0:2]), axis=-1, keepdims=True), 0.0, rtol=0.0, atol=1.0e-16),
                      np.block([np.zeros_like(XYZ[..., 0:2]), np.sqrt(6.0 / np.pi) * XYZ[..., 2:3]]),
                      np.block([np.where(order, T[..., 0:1], T[..., 1:2]) * q,
                                np.where(order, T[..., 1:2], T[..., 0:1]) * q,
                                np.sqrt(6.0 / np.pi) * XYZ[..., 2:3] - c])
                      )

    ho[np.isclose(np.sum(np.abs(cu), axis=-1), 0.0, rtol=0.0, atol=1.0e-16)] = 0.0
    ho = np.take_along_axis(ho, _get_pyramid_order(cu, 'backward'), -1)

    # return ho # here for homochoric

    """Step 2: Homochoric vector to axis angle pair."""
    tfit = np.array([+1.0000000000018852, -0.5000000002194847,
                     -0.024999992127593126, -0.003928701544781374,
                     -0.0008152701535450438, -0.0002009500426119712,
                     -0.00002397986776071756, -0.00008202868926605841,
                     +0.00012448715042090092, -0.0001749114214822577,
                     +0.0001703481934140054, -0.00012062065004116828,
                     +0.000059719705868660826, -0.00001980756723965647,
                     +0.000003953714684212874, -0.00000036555001439719544])
    hmag_squared = np.sum(ho ** 2., axis=-1, keepdims=True)
    hm = hmag_squared.copy()
    s = tfit[0] + tfit[1] * hmag_squared
    for i in range(2, 16):
        hm *= hmag_squared
        s += tfit[i] * hm
    with np.errstate(invalid='ignore'):
        ax = np.where(np.broadcast_to(np.abs(hmag_squared) < 1.e-8, ho.shape[:-1] + (4,)),
                      [0.0, 0.0, 1.0, 0.0],
                      np.block([ho / np.sqrt(hmag_squared), 2.0 * np.arccos(np.clip(s, -1.0, 1.0))]))
    # return ax # here for axis angle pair

    """Step 3: Axis angle pair to quaternion."""
    c = np.cos(ax[..., 3:4] * .5)
    s = np.sin(ax[..., 3:4] * .5)
    qu = np.where(np.abs(ax[..., 3:4]) < 1.e-6, [1.0, 0.0, 0.0, 0.0], np.block([c, ax[..., :3] * s]))

    # Converting Quaternion Order convention
    # DREAM3D convention is <xyz> s (<xyz> = imaginary vector component)
    # Convention used here is s <xyz>
    # This conversion is s,<xyz> to <xyz>,s
    if scalar_first is not True:
        tempqu = np.transpose(np.asarray([qu[:, 1], qu[:, 2], qu[:, 3], qu[:, 0]]))
        qu = tempqu
    #-------

    return qu

def cubo2rod(cu):
    """Cubochoric vector to quaternion."""

    """
        Step 1: Cubochoric vector to homochoric vector.

        References
        ----------
        D. Roşca et al., Modelling and Simulation in Materials Science and Engineering 22:075013, 2014
        https://doi.org/10.1088/0965-0393/22/7/075013

        """
    with np.errstate(invalid='ignore', divide='ignore'):
        # get pyramide and scale by grid parameter ratio
        XYZ = np.take_along_axis(cu, _get_pyramid_order(cu, 'forward'), -1) * sc
        order = np.abs(XYZ[..., 1:2]) <= np.abs(XYZ[..., 0:1])
        q = np.pi / 12.0 * np.where(order, XYZ[..., 1:2], XYZ[..., 0:1]) \
            / np.where(order, XYZ[..., 0:1], XYZ[..., 1:2])
        c = np.cos(q)
        s = np.sin(q)
        q = R1 * 2.0 ** 0.25 / beta / np.sqrt(np.sqrt(2.0) - c) \
            * np.where(order, XYZ[..., 0:1], XYZ[..., 1:2])

        T = np.block([(np.sqrt(2.0) * c - 1.0), np.sqrt(2.0) * s]) * q

        # transform to sphere grid (inverse Lambert)
        c = np.sum(T ** 2, axis=-1, keepdims=True)
        s = c * np.pi / 24.0 / XYZ[..., 2:3] ** 2
        c = c * np.sqrt(np.pi / 24.0) / XYZ[..., 2:3]
        q = np.sqrt(1.0 - s)

        ho = np.where(np.isclose(np.sum(np.abs(XYZ[..., 0:2]), axis=-1, keepdims=True), 0.0, rtol=0.0, atol=1.0e-16),
                      np.block([np.zeros_like(XYZ[..., 0:2]), np.sqrt(6.0 / np.pi) * XYZ[..., 2:3]]),
                      np.block([np.where(order, T[..., 0:1], T[..., 1:2]) * q,
                                np.where(order, T[..., 1:2], T[..., 0:1]) * q,
                                np.sqrt(6.0 / np.pi) * XYZ[..., 2:3] - c])
                      )

    ho[np.isclose(np.sum(np.abs(cu), axis=-1), 0.0, rtol=0.0, atol=1.0e-16)] = 0.0
    ho = np.take_along_axis(ho, _get_pyramid_order(cu, 'backward'), -1)

    # return ho # here for homochoric

    """Step 2: Homochoric vector to axis angle pair."""
    tfit = np.array([+1.0000000000018852, -0.5000000002194847,
                     -0.024999992127593126, -0.003928701544781374,
                     -0.0008152701535450438, -0.0002009500426119712,
                     -0.00002397986776071756, -0.00008202868926605841,
                     +0.00012448715042090092, -0.0001749114214822577,
                     +0.0001703481934140054, -0.00012062065004116828,
                     +0.000059719705868660826, -0.00001980756723965647,
                     +0.000003953714684212874, -0.00000036555001439719544])
    hmag_squared = np.sum(ho ** 2., axis=-1, keepdims=True)
    hm = hmag_squared.copy()
    s = tfit[0] + tfit[1] * hmag_squared
    for i in range(2, 16):
        hm *= hmag_squared
        s += tfit[i] * hm
    with np.errstate(invalid='ignore'):
        ax = np.where(np.broadcast_to(np.abs(hmag_squared) < 1.e-8, ho.shape[:-1] + (4,)),
                      [0.0, 0.0, 1.0, 0.0],
                      np.block([ho / np.sqrt(hmag_squared), 2.0 * np.arccos(np.clip(s, -1.0, 1.0))]))
    # return ax # here for axis angle pair

    """Step 3: Axis angle pair to Rodrigues-Frank vector."""
    ro = np.block([ax[...,:3],
                   np.where(np.isclose(ax[...,3:4],np.pi,atol=1.e-15,rtol=.0),
                            np.inf,
                            np.tan(ax[...,3:4]*0.5))
                  ])
    ro[np.abs(ax[...,3])<1.e-6] = [.0,.0,P,.0]

    return ro

#---------- Rodrigues Operations ----------

def rod2cubo(ro):
    """ Step 1: Rodrigues-Frank vector to homochoric vector."""

    f = np.where(np.isfinite(ro[...,3:4]),2.0*np.arctan(ro[...,3:4]) -np.sin(2.0*np.arctan(ro[...,3:4])),np.pi)
    ho = np.where(np.broadcast_to(np.sum(ro[...,0:3]**2.0,axis=-1,keepdims=True) < 1.e-8,ro[...,0:3].shape),
                  np.zeros(3), ro[...,0:3]* np.cbrt(0.75*f))

    # return ho # here for homochoric vector

    """
        Step 2: Homochoric vector to cubochoric vector.

        References
        ----------
        D. Roşca et al., Modelling and Simulation in Materials Science and Engineering 22:075013, 2014
        https://doi.org/10.1088/0965-0393/22/7/075013

        """
    rs = np.linalg.norm(ho, axis=-1, keepdims=True)

    xyz3 = np.take_along_axis(ho, _get_pyramid_order(ho, 'forward'), -1)

    with np.errstate(invalid='ignore', divide='ignore'):
        # inverse M_3
        xyz2 = xyz3[..., 0:2] * np.sqrt(2.0 * rs / (rs + np.abs(xyz3[..., 2:3])))
        qxy = np.sum(xyz2 ** 2, axis=-1, keepdims=True)

        q2 = qxy + np.max(np.abs(xyz2), axis=-1, keepdims=True) ** 2
        sq2 = np.sqrt(q2)
        q = (beta / np.sqrt(2.0) / R1) * np.sqrt(q2 * qxy / (q2 - np.max(np.abs(xyz2), axis=-1, keepdims=True) * sq2))
        tt = np.clip((np.min(np.abs(xyz2), axis=-1, keepdims=True) ** 2 \
                      + np.max(np.abs(xyz2), axis=-1, keepdims=True) * sq2) / np.sqrt(2.0) / qxy, -1.0, 1.0)
        T_inv = np.where(np.abs(xyz2[..., 1:2]) <= np.abs(xyz2[..., 0:1]),
                         np.block([np.ones_like(tt), np.arccos(tt) / np.pi * 12.0]),
                         np.block([np.arccos(tt) / np.pi * 12.0, np.ones_like(tt)])) * q
        T_inv[xyz2 < 0.0] *= -1.0
        T_inv[np.broadcast_to(np.isclose(qxy, 0.0, rtol=0.0, atol=1.0e-12), T_inv.shape)] = 0.0
        cu = np.block(
            [T_inv, np.where(xyz3[..., 2:3] < 0.0, -np.ones_like(xyz3[..., 2:3]), np.ones_like(xyz3[..., 2:3])) \
             * rs / np.sqrt(6.0 / np.pi),
             ]) / sc

    cu[np.isclose(np.sum(np.abs(ho), axis=-1), 0.0, rtol=0.0, atol=1.0e-16)] = 0.0
    cu = np.take_along_axis(cu, _get_pyramid_order(ho, 'backward'), -1)

    return cu

def rod2quat(ro, scalar_first=True):
    """Step 1:  Rodrigues-Frank vector to axis angle pair."""
    with np.errstate(invalid='ignore',divide='ignore'):
        ax = np.where(np.isfinite(ro[...,3:4]),
             np.block([ro[...,0:3]*np.linalg.norm(ro[...,0:3],axis=-1,keepdims=True),2.*np.arctan(ro[...,3:4])]),
             np.block([ro[...,0:3],np.broadcast_to(np.pi,ro[...,3:4].shape)]))
    ax[np.abs(ro[...,3]) < 1.e-8]  = np.array([ 0.0, 0.0, 1.0, 0.0 ])
    # return ax # here for axis angle pair

    """Step 2: Axis angle pair to quaternion."""
    c = np.cos(ax[...,3:4]*.5)
    s = np.sin(ax[...,3:4]*.5)
    qu = np.where(np.abs(ax[...,3:4])<1.e-6,[1.0, 0.0, 0.0, 0.0],np.block([c, ax[...,:3]*s]))

    # Converting Quaternion Order convention
    # DREAM3D convention is <xyz> s (<xyz> = imaginary vector component)
    # Convention used here is s <xyz>
    # This conversion is s,<xyz> to <xyz>,s
    if scalar_first is not True:
        tempqu = np.transpose(np.asarray([qu[:, 1], qu[:, 2], qu[:, 3], qu[:, 0]]))
        qu = tempqu
    # -------

    return qu

#---------- Cubochoric Convention Functions ----------

def _get_pyramid_order(xyz,direction=None):
    """
    Get order of the coordinates.

    Depending on the pyramid in which the point is located, the order need to be adjusted.

    Parameters
    ----------
    xyz : numpy.ndarray
       coordinates of a point on a uniform refinable grid on a ball or
       in a uniform refinable cubical grid.

    References
    ----------
    D. Roşca et al., Modelling and Simulation in Materials Science and Engineering 22:075013, 2014
    https://doi.org/10.1088/0965-0393/22/7/075013

    """
    order = {'forward': np.array([[0,1,2],[1,2,0],[2,0,1]]),
             'backward':np.array([[0,1,2],[2,0,1],[1,2,0]])}

    p = np.where(np.maximum(np.abs(xyz[...,0]),np.abs(xyz[...,1])) <= np.abs(xyz[...,2]),0,
                 np.where(np.maximum(np.abs(xyz[...,1]),np.abs(xyz[...,2])) <= np.abs(xyz[...,0]),1,2))

    return order[direction][p]
