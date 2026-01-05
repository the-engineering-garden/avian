//! Tree acceleration structures for spatial queries on [colliders](crate::collision::collider::Collider).
//!
//! To speed up [broad phase](crate::collision::broad_phase) collision detection and [spatial queries](crate::spatial_query),
//! Avian maintains a [`ColliderTree`] structure for all colliders in the physics world. This is implemented as
//! a [Bounding Volume Hierarchy (BVH)][BVH], which accelerates querying for [AABB](crate::collision::collider::ColliderAabb)
//! overlaps, ray intersections, and more.
//!
//! Colliders of dynamic, kinematic, and static bodies are all stored in a separate [`ColliderTree`](ColliderTree)
//! to allow efficiently querying for specific subsets of colliders and to optimize tree updates based on body type.
//! Trees for dynamic and kinematic bodies are rebuilt every physics step, while the static tree is incrementally updated
//! when static colliders are added, removed, or modified. The trees are stored in the [`ColliderTrees`] resource.
//!
//! [BVH]: https://en.wikipedia.org/wiki/Bounding_volume_hierarchy
//!
//! # Usage
//!
//! The collider trees are fairly low-level, and not intended to be used directly.
//! For spatial queries, consider using the higher-level [`SpatialQuery`] API instead,
//! and for broad phase collision detection, consider using the [`BvhBroadPhasePlugin`].
//!
//! [`SpatialQuery`]: crate::spatial_query::SpatialQuery
//! [`BvhBroadPhasePlugin`]: crate::collision::broad_phase::bvh::BvhBroadPhasePlugin

mod optimization;
mod tree;
mod update;

pub use optimization::{ColliderTreeOptimization, TreeOptimizationMode};
pub use tree::{ColliderTree, ColliderTreeProxy, ColliderTreeProxyFlags, ColliderTreeWorkspace};
pub use update::{ColliderTreeProxyIndex, MovedProxies};

use optimization::ColliderTreeOptimizationPlugin;
use update::ColliderTreeUpdatePlugin;

use core::marker::PhantomData;

use crate::prelude::*;
use bevy::prelude::*;

/// A plugin that manages [`ColliderTrees`] for a collider type `C`.
pub struct ColliderTreePlugin<C: AnyCollider>(PhantomData<C>);

impl<C: AnyCollider> Default for ColliderTreePlugin<C> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<C: AnyCollider> Plugin for ColliderTreePlugin<C> {
    fn build(&self, app: &mut App) {
        // Add plugin for updating trees as colliders move.
        app.add_plugins(ColliderTreeUpdatePlugin::<C>::default());

        // Add plugin for optimizing trees tp maintain good query performance.
        if !app.is_plugin_added::<ColliderTreeOptimizationPlugin>() {
            app.add_plugins(ColliderTreeOptimizationPlugin);
        }

        // Initialize resources.
        app.init_resource::<ColliderTrees>()
            .init_resource::<MovedProxies>();

        // Configure system sets.
        app.configure_sets(
            PhysicsSchedule,
            ColliderTreeSystems::UpdateAabbs
                .in_set(PhysicsStepSystems::BroadPhase)
                .after(BroadPhaseSystems::First)
                .before(BroadPhaseSystems::CollectCollisions),
        );
        app.configure_sets(
            PhysicsSchedule,
            ColliderTreeSystems::BeginOptimize.in_set(NarrowPhaseSystems::Update),
        );
        app.configure_sets(
            PhysicsSchedule,
            ColliderTreeSystems::EndOptimize.in_set(PhysicsStepSystems::Finalize),
        );
    }
}

/// System sets for managing [`ColliderTrees`].
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ColliderTreeSystems {
    /// Updates the AABBs of colliders.
    UpdateAabbs,
    /// Begins optimizing acceleration structures to keep their query performance good.
    ///
    /// This runs concurrently with the simulation step as an async task.
    BeginOptimize,
    /// Completes the optimization of acceleration structures started in [`ColliderTreeSystems::BeginOptimize`].
    ///
    /// This runs at the end of the simulation step.
    EndOptimize,
}

/// Trees for accelerating queries on a set of colliders.
#[derive(Resource, Default)]
pub struct ColliderTrees {
    /// A tree for the colliders of dynamic and kinematic bodies.
    pub dynamic_tree: ColliderTree,
    /// A tree for the colliders of static bodies.
    pub static_tree: ColliderTree,
}
