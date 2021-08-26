Basic Mechanica Types
^^^^^^^^^^^^^^^^^^^^^^

Mechanica uses some basic types that provide support and convenience
methods for particular operations, especially concerning vector and
tensor operations. Some of these types are completely native to
Mechanica, and others constructed partially or completely from
types distributed in various Mechanica dependencies (*e.g.*,
:class:`MxVector3f` from :class:`Vector3`, from
`GLFW <https://www.glfw.org/>`_.

.. autoclass:: MxVector2d

    A 2D vector with ``double`` elements

    .. automethod:: xAxis

    .. automethod:: yAxis

    .. automethod:: xScale

    .. automethod:: yScale

    .. automethod:: x

    .. automethod:: y

    .. automethod:: flipped

    .. automethod:: dot

    .. automethod:: __len__

    .. automethod:: __iadd__

    .. automethod:: __add__

    .. automethod:: __isub__

    .. automethod:: __sub__

    .. automethod:: __imul__

    .. automethod:: __mul__

    .. automethod:: __itruediv__

    .. automethod:: __truediv__

    .. automethod:: length

    .. automethod:: normalized

    .. automethod:: resized

    .. automethod:: projected

    .. automethod:: projectedOntoNormalized

    .. automethod:: distance

    .. automethod:: __getitem__

    .. automethod:: __setitem__

    .. automethod:: as_list

.. autoclass:: MxVector2f

    A 2D vector with ``float`` elements

    .. automethod:: xAxis

    .. automethod:: yAxis

    .. automethod:: xScale

    .. automethod:: yScale

    .. automethod:: x

    .. automethod:: y

    .. automethod:: flipped

    .. automethod:: dot

    .. automethod:: __len__

    .. automethod:: __iadd__

    .. automethod:: __add__

    .. automethod:: __isub__

    .. automethod:: __sub__

    .. automethod:: __imul__

    .. automethod:: __mul__

    .. automethod:: __itruediv__

    .. automethod:: __truediv__

    .. automethod:: length

    .. automethod:: normalized

    .. automethod:: resized

    .. automethod:: projected

    .. automethod:: projectedOntoNormalized

    .. automethod:: distance

    .. automethod:: __getitem__

    .. automethod:: __setitem__

    .. automethod:: as_list

.. autoclass:: MxVector2i

    A 2D vector with ``int`` elements

    .. automethod:: xAxis

    .. automethod:: yAxis

    .. automethod:: xScale

    .. automethod:: yScale

    .. automethod:: x

    .. automethod:: y

    .. automethod:: flipped

    .. automethod:: dot

    .. automethod:: __len__

    .. automethod:: __iadd__

    .. automethod:: __add__

    .. automethod:: __isub__

    .. automethod:: __sub__

    .. automethod:: __imul__

    .. automethod:: __mul__

    .. automethod:: __itruediv__

    .. automethod:: __truediv__

    .. automethod:: __getitem__

    .. automethod:: __setitem__

    .. automethod:: as_list

.. autoclass:: MxVector3d

    A 3D vector with ``double`` elements

    .. automethod:: xAxis

    .. automethod:: yAxis

    .. automethod:: zAxis

    .. automethod:: xScale

    .. automethod:: yScale

    .. automethod:: zScale

    .. automethod:: x

    .. automethod:: y

    .. automethod:: z

    .. automethod:: r

    .. automethod:: g

    .. automethod:: b

    .. automethod:: xy

    .. automethod:: flipped

    .. automethod:: dot

    .. automethod:: __len__

    .. automethod:: __iadd__

    .. automethod:: __add__

    .. automethod:: __isub__

    .. automethod:: __sub__

    .. automethod:: __imul__

    .. automethod:: __mul__

    .. automethod:: __itruediv__

    .. automethod:: __truediv__

    .. automethod:: length

    .. automethod:: normalized

    .. automethod:: resized

    .. automethod:: projected

    .. automethod:: projectedOntoNormalized

    .. automethod:: distance

    .. automethod:: __getitem__

    .. automethod:: __setitem__

    .. automethod:: as_list

.. autoclass:: MxVector3f

    A 3D vector with ``float`` elements

    .. automethod:: xAxis

    .. automethod:: yAxis

    .. automethod:: zAxis

    .. automethod:: xScale

    .. automethod:: yScale

    .. automethod:: zScale

    .. automethod:: x

    .. automethod:: y

    .. automethod:: z

    .. automethod:: r

    .. automethod:: g

    .. automethod:: b

    .. automethod:: xy

    .. automethod:: flipped

    .. automethod:: dot

    .. automethod:: __len__

    .. automethod:: __iadd__

    .. automethod:: __add__

    .. automethod:: __isub__

    .. automethod:: __sub__

    .. automethod:: __imul__

    .. automethod:: __mul__

    .. automethod:: __itruediv__

    .. automethod:: __truediv__

    .. automethod:: length

    .. automethod:: normalized

    .. automethod:: resized

    .. automethod:: projected

    .. automethod:: projectedOntoNormalized

    .. automethod:: distance

    .. automethod:: __getitem__

    .. automethod:: __setitem__

    .. automethod:: as_list

.. autoclass:: MxVector3i

    A 3D vector with ``int`` elements

    .. automethod:: xAxis

    .. automethod:: yAxis

    .. automethod:: zAxis

    .. automethod:: xScale

    .. automethod:: yScale

    .. automethod:: zScale

    .. automethod:: x

    .. automethod:: y

    .. automethod:: z

    .. automethod:: r

    .. automethod:: g

    .. automethod:: b

    .. automethod:: xy

    .. automethod:: flipped

    .. automethod:: dot

    .. automethod:: __len__

    .. automethod:: __iadd__

    .. automethod:: __add__

    .. automethod:: __isub__

    .. automethod:: __sub__

    .. automethod:: __imul__

    .. automethod:: __mul__

    .. automethod:: __itruediv__

    .. automethod:: __truediv__

    .. automethod:: __getitem__

    .. automethod:: __setitem__

    .. automethod:: as_list

.. autoclass:: MxVector4d

    A 4D vector with ``double`` elements

    .. automethod:: x

    .. automethod:: y

    .. automethod:: z

    .. automethod:: w

    .. automethod:: r

    .. automethod:: g

    .. automethod:: b

    .. automethod:: a

    .. automethod:: xyz

    .. automethod:: rgb

    .. automethod:: xy

    .. automethod:: flipped

    .. automethod:: dot

    .. automethod:: __len__

    .. automethod:: __iadd__

    .. automethod:: __add__

    .. automethod:: __isub__

    .. automethod:: __sub__

    .. automethod:: __imul__

    .. automethod:: __mul__

    .. automethod:: __itruediv__

    .. automethod:: __truediv__

    .. automethod:: length

    .. automethod:: normalized

    .. automethod:: resized

    .. automethod:: projected

    .. automethod:: projectedOntoNormalized

    .. automethod:: distance

    .. automethod:: distanceScaled

    .. automethod:: planeEquation

    .. automethod:: __getitem__

    .. automethod:: __setitem__

    .. automethod:: as_list

.. autoclass:: MxVector4f

    A 4D vector with ``float`` elements

    .. automethod:: x

    .. automethod:: y

    .. automethod:: z

    .. automethod:: w

    .. automethod:: r

    .. automethod:: g

    .. automethod:: b

    .. automethod:: a

    .. automethod:: xyz

    .. automethod:: rgb

    .. automethod:: xy

    .. automethod:: flipped

    .. automethod:: dot

    .. automethod:: __len__

    .. automethod:: __iadd__

    .. automethod:: __add__

    .. automethod:: __isub__

    .. automethod:: __sub__

    .. automethod:: __imul__

    .. automethod:: __mul__

    .. automethod:: __itruediv__

    .. automethod:: __truediv__

    .. automethod:: length

    .. automethod:: normalized

    .. automethod:: resized

    .. automethod:: projected

    .. automethod:: projectedOntoNormalized

    .. automethod:: distance

    .. automethod:: distanceScaled

    .. automethod:: planeEquation

    .. automethod:: __getitem__

    .. automethod:: __setitem__

    .. automethod:: as_list

.. autoclass:: MxVector4i

    A 4D vector with ``int`` elements

    .. automethod:: x

    .. automethod:: y

    .. automethod:: z

    .. automethod:: w

    .. automethod:: r

    .. automethod:: g

    .. automethod:: b

    .. automethod:: a

    .. automethod:: xyz

    .. automethod:: rgb

    .. automethod:: xy

    .. automethod:: flipped

    .. automethod:: dot

    .. automethod:: __len__

    .. automethod:: __iadd__

    .. automethod:: __add__

    .. automethod:: __isub__

    .. automethod:: __sub__

    .. automethod:: __imul__

    .. automethod:: __mul__

    .. automethod:: __itruediv__

    .. automethod:: __truediv__

    .. automethod:: __getitem__

    .. automethod:: __setitem__

    .. automethod:: as_list

.. autoclass:: MxMatrix3d

    A 3x3 square matrix with ``double`` elements

    .. automethod:: rotation

    .. automethod:: shearingX

    .. automethod:: shearingY

    .. automethod:: isRigidTransformation

    .. automethod:: invertedRigid

    .. automethod:: __neg__

    .. automethod:: __iadd__

    .. automethod:: __add__

    .. automethod:: __isub__

    .. automethod:: __sub__

    .. automethod:: __imul__

    .. automethod:: __mul__

    .. automethod:: __itruediv__

    .. automethod:: __truediv__

    .. automethod:: flippedCols

    .. automethod:: flippedRows

    .. automethod:: row

    .. automethod:: __mul__

    .. automethod:: transposed

    .. automethod:: diagonal

    .. automethod:: inverted

    .. automethod:: invertedOrthogonal

    .. automethod:: __len__

    .. automethod:: __getitem__

    .. automethod:: __setitem__

    .. automethod:: as_lists

.. autoclass:: MxMatrix3f

    A 3x3 square matrix with ``float`` elements

    .. automethod:: rotation

    .. automethod:: shearingX

    .. automethod:: shearingY

    .. automethod:: isRigidTransformation

    .. automethod:: invertedRigid

    .. automethod:: __neg__

    .. automethod:: __iadd__

    .. automethod:: __add__

    .. automethod:: __isub__

    .. automethod:: __sub__

    .. automethod:: __imul__

    .. automethod:: __mul__

    .. automethod:: __itruediv__

    .. automethod:: __truediv__

    .. automethod:: flippedCols

    .. automethod:: flippedRows

    .. automethod:: row

    .. automethod:: __mul__

    .. automethod:: transposed

    .. automethod:: diagonal

    .. automethod:: inverted

    .. automethod:: invertedOrthogonal

    .. automethod:: __len__

    .. automethod:: __getitem__

    .. automethod:: __setitem__

    .. automethod:: as_lists

.. autoclass:: MxMatrix4d

    A 4x4 square matrix with ``double`` elements

    .. automethod:: rotationX

    .. automethod:: rotationY

    .. automethod:: rotationZ

    .. automethod:: reflection

    .. automethod:: shearingXY

    .. automethod:: shearingXZ

    .. automethod:: shearingYZ

    .. automethod:: orthographicProjection

    .. automethod:: perspectiveProjection

    .. automethod:: lookAt

    .. automethod:: isRigidTransformation

    .. automethod:: rotationScaling

    .. automethod:: rotationShear

    .. automethod:: rotation

    .. automethod:: rotationNormalized

    .. automethod:: scalingSquared

    .. automethod:: scaling

    .. automethod:: uniformScalingSquared

    .. automethod:: uniformScaling

    .. automethod:: normalMatrix

    .. automethod:: right

    .. automethod:: up

    .. automethod:: backward

    .. automethod:: translation

    .. automethod:: invertedRigid

    .. automethod:: transformVector

    .. automethod:: transformPoint

    .. automethod:: __neg__

    .. automethod:: __iadd__

    .. automethod:: __add__

    .. automethod:: __isub__

    .. automethod:: __sub__

    .. automethod:: __imul__

    .. automethod:: __itruediv__

    .. automethod:: __truediv__

    .. automethod:: flippedCols

    .. automethod:: flippedRows

    .. automethod:: row

    .. automethod:: __mul__

    .. automethod:: transposed

    .. automethod:: diagonal

    .. automethod:: inverted

    .. automethod:: invertedOrthogonal

    .. automethod:: __len__

    .. automethod:: __getitem__

    .. automethod:: __setitem__

    .. automethod:: as_lists

.. autoclass:: MxMatrix4f

    A 4x4 square matrix with ``float`` elements

    .. automethod:: rotationX

    .. automethod:: rotationY

    .. automethod:: rotationZ

    .. automethod:: reflection

    .. automethod:: shearingXY

    .. automethod:: shearingXZ

    .. automethod:: shearingYZ

    .. automethod:: orthographicProjection

    .. automethod:: perspectiveProjection

    .. automethod:: lookAt

    .. automethod:: isRigidTransformation

    .. automethod:: rotationScaling

    .. automethod:: rotationShear

    .. automethod:: rotation

    .. automethod:: rotationNormalized

    .. automethod:: scalingSquared

    .. automethod:: scaling

    .. automethod:: uniformScalingSquared

    .. automethod:: uniformScaling

    .. automethod:: normalMatrix

    .. automethod:: right

    .. automethod:: up

    .. automethod:: backward

    .. automethod:: translation

    .. automethod:: invertedRigid

    .. automethod:: transformVector

    .. automethod:: transformPoint

    .. automethod:: __neg__

    .. automethod:: __iadd__

    .. automethod:: __add__

    .. automethod:: __isub__

    .. automethod:: __sub__

    .. automethod:: __imul__

    .. automethod:: __itruediv__

    .. automethod:: __truediv__

    .. automethod:: flippedCols

    .. automethod:: flippedRows

    .. automethod:: row

    .. automethod:: __mul__

    .. automethod:: transposed

    .. automethod:: diagonal

    .. automethod:: inverted

    .. automethod:: invertedOrthogonal

    .. automethod:: __len__

    .. automethod:: __getitem__

    .. automethod:: __setitem__

    .. automethod:: as_lists

.. autoclass:: MxQuaterniond

    A quaternion with ``double`` elements

    .. automethod:: rotation

    .. automethod:: fromMatrix

    .. automethod:: data

    .. automethod:: __eq__

    .. automethod:: __ne__

    .. automethod:: isNormalized

    .. automethod:: vector

    .. automethod:: scalar

    .. automethod:: angle

    .. automethod:: axis

    .. automethod:: toMatrix

    .. automethod:: toEuler

    .. automethod:: __neg__

    .. automethod:: __iadd__

    .. automethod:: __add__

    .. automethod:: __isub__

    .. automethod:: __sub__

    .. automethod:: __imul__

    .. automethod:: __itruediv__

    .. automethod:: __truediv__

    .. automethod:: __mul__

    .. automethod:: dot

    .. automethod:: length

    .. automethod:: normalized

    .. automethod:: conjugated

    .. automethod:: inverted

    .. automethod:: invertedNormalized

    .. automethod:: transformVector

    .. automethod:: transformVectorNormalized

    .. automethod:: __getitem__

    .. automethod:: __setitem__

    .. automethod:: as_list

.. autoclass:: MxQuaternionf

    A quaternion with ``float`` elements

    .. automethod:: rotation

    .. automethod:: fromMatrix

    .. automethod:: data

    .. automethod:: __eq__

    .. automethod:: __ne__

    .. automethod:: isNormalized

    .. automethod:: vector

    .. automethod:: scalar

    .. automethod:: angle

    .. automethod:: axis

    .. automethod:: toMatrix

    .. automethod:: toEuler

    .. automethod:: __neg__

    .. automethod:: __iadd__

    .. automethod:: __add__

    .. automethod:: __isub__

    .. automethod:: __sub__

    .. automethod:: __imul__

    .. automethod:: __itruediv__

    .. automethod:: __truediv__

    .. automethod:: __mul__

    .. automethod:: dot

    .. automethod:: length

    .. automethod:: normalized

    .. automethod:: conjugated

    .. automethod:: inverted

    .. automethod:: invertedNormalized

    .. automethod:: transformVector

    .. automethod:: transformVectorNormalized

    .. automethod:: __getitem__

    .. automethod:: __setitem__

    .. automethod:: as_list
