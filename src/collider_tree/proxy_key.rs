use core::hint::unreachable_unchecked;

use bevy::{ecs::component::Component, reflect::Reflect};

use crate::prelude::RigidBody;

/// A key for a proxy in a [`ColliderTree`], encoding both
/// the [`ProxyId`] and the tree type (dynamic, kinematic, static).
///
/// The tree type is stored in the lower 2 bits of the key,
/// leaving 30 bits for the [`ProxyId`].
///
/// [`ColliderTree`]: crate::collider_tree::ColliderTree
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq, Hash, Reflect)]
pub struct ColliderTreeProxyKey(u32);

impl ColliderTreeProxyKey {
    /// A placeholder proxy key used before the proxy is actually created.
    pub const PLACEHOLDER: Self = ColliderTreeProxyKey(u32::MAX);

    /// Creates a new [`ColliderTreeProxyKey`] from the given [`ProxyId`] and tree type.
    #[inline]
    pub const fn new(id: ProxyId, body: RigidBody) -> Self {
        // Encode the tree type in the lower 2 bits.
        ColliderTreeProxyKey((id.id() << 2) | body as u32)
    }

    /// Returns the [`ProxyId`] of the proxy.
    #[inline]
    pub const fn id(&self) -> ProxyId {
        ProxyId::new(self.0 >> 2)
    }

    /// Returns the tree type.
    #[inline]
    pub const fn body(&self) -> RigidBody {
        match self.0 & 0b11 {
            // TODO: The "dynamic, static, kinematic" order is a bit weird,
            //       but it comes from the order of the `RigidBody` enum.
            //       Consider changing it in the future.
            0 => RigidBody::Dynamic,
            1 => RigidBody::Static,
            2 => RigidBody::Kinematic,
            // Safety: Bitwise AND with 0b11 can only yield 0, 1, or 2.
            _ => unsafe { unreachable_unchecked() },
        }
    }

    /// Returns `true` if the proxy belongs to a dynamic body.
    #[inline]
    pub const fn is_dynamic(&self) -> bool {
        self.body() as u32 == RigidBody::Dynamic as u32
    }

    /// Returns `true` if the proxy belongs to a static body.
    #[inline]
    pub const fn is_static(&self) -> bool {
        self.body() as u32 == RigidBody::Static as u32
    }

    /// Returns `true` if the proxy belongs to a kinematic body.
    #[inline]
    pub const fn is_kinematic(&self) -> bool {
        self.body() as u32 == RigidBody::Kinematic as u32
    }
}

/// A stable identifier for a proxy in a [`ColliderTree`].
///
/// [`ColliderTree`]: crate::collider_tree::ColliderTree
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Reflect)]
pub struct ProxyId(u32);

impl ProxyId {
    /// A placeholder proxy ID used before the proxy is actually created.
    pub const PLACEHOLDER: Self = ProxyId(u32::MAX >> 2);

    /// Creates a new [`ProxyId`] from the given `u32` identifier.
    ///
    /// Only the lower 30 bits should be used for the ID.
    ///
    /// # Panics
    ///
    /// Panics if either of the upper 2 bits are set and `debug_assertions` are enabled.
    #[inline]
    pub const fn new(id: u32) -> Self {
        debug_assert!(id < (1 << 30), "ProxyId can only use lower 30 bits");
        ProxyId(id)
    }

    /// Returns the proxy ID as a `u32`.
    #[inline]
    pub const fn id(&self) -> u32 {
        self.0
    }

    /// Returns the proxy ID as a `usize`.
    #[inline]
    pub const fn index(&self) -> usize {
        self.0 as usize
    }
}

impl From<u32> for ProxyId {
    #[inline]
    fn from(id: u32) -> Self {
        ProxyId::new(id)
    }
}

impl From<ProxyId> for u32 {
    #[inline]
    fn from(proxy_id: ProxyId) -> Self {
        proxy_id.id()
    }
}

impl PartialOrd for ProxyId {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ProxyId {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}
