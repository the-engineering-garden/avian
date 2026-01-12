#[cfg(feature = "parallel")]
use core::cell::RefCell;
use core::marker::PhantomData;

use crate::{
    collider_tree::{
        ColliderTreeProxy, ColliderTreeProxyKey, ColliderTreeSystems, ColliderTrees, ProxyId,
        tree::ColliderTreeProxyFlags,
    },
    collision::{broad_phase::BroadPhaseDiagnostics, collider::EnlargedAabb},
    data_structures::bit_vec::BitVec,
    dynamics::solver::solver_body::SolverBody,
    prelude::*,
};
use bevy::{ecs::system::StaticSystemParam, platform::collections::HashSet, prelude::*};
use obvhs::aabb::Aabb;
#[cfg(feature = "parallel")]
use thread_local::ThreadLocal;

/// A plugin for updating [`ColliderTree`]s for a collider type `C`.
pub(super) struct ColliderTreeUpdatePlugin<C: AnyCollider>(PhantomData<C>);

impl<C: AnyCollider> Default for ColliderTreeUpdatePlugin<C> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<C: AnyCollider> Plugin for ColliderTreeUpdatePlugin<C> {
    fn build(&self, app: &mut App) {
        // Initialize resources.
        app.init_resource::<MovedProxies>();

        // Add systems for updating collider AABBs.
        app.add_systems(
            PhysicsSchedule,
            (update_dynamic_aabbs::<C>, update_static_aabbs::<C>)
                .chain()
                .in_set(ColliderTreeSystems::UpdateAabbs)
                // Allowing ambiguities is required so that it's possible
                // to have multiple collision backends at the same time.
                .ambiguous_with_all(),
        );

        // Initialize `ColliderAabb` for colliders.
        app.add_observer(
            |trigger: On<Add, C>,
             mut query: Query<(
                &C,
                &Position,
                &Rotation,
                Option<&CollisionMargin>,
                &mut ColliderAabb,
            )>,
             narrow_phase_config: Res<NarrowPhaseConfig>,
             length_unit: Res<PhysicsLengthUnit>,
             collider_context: StaticSystemParam<C::Context>| {
                let contact_tolerance = length_unit.0 * narrow_phase_config.contact_tolerance;
                let aabb_context = AabbContext::new(trigger.entity, &*collider_context);

                if let Ok((collider, pos, rot, collision_margin, mut aabb)) =
                    query.get_mut(trigger.entity)
                {
                    let collision_margin = collision_margin.map_or(0.0, |m| m.0);
                    *aabb = collider
                        .aabb_with_context(pos.0, *rot, aabb_context)
                        .grow(Vector::splat(contact_tolerance + collision_margin));
                }
            },
        );

        app.add_observer(
            |trigger: On<Add, ColliderOf>,
             body_query: Query<&RigidBody>,
             mut collider_query: Query<(
                &ColliderOf,
                &ColliderAabb,
                &EnlargedAabb,
                &mut ColliderTreeProxyKey,
                Option<&CollisionLayers>,
            )>,
             mut trees: ResMut<ColliderTrees>| {
                let entity = trigger.entity;

                let Ok((collider_of, collider_aabb, enlarged_aabb, mut proxy_key, layers)) =
                    collider_query.get_mut(entity)
                else {
                    return;
                };
                let Ok(rb) = body_query.get(collider_of.body) else {
                    return;
                };

                let aabb = Aabb::from(*collider_aabb);
                let enlarged_aabb = Aabb::from(enlarged_aabb.get());

                let proxy = ColliderTreeProxy {
                    entity,
                    body: collider_of.body,
                    layers: layers.copied().unwrap_or_default(),
                    aabb,
                    flags: ColliderTreeProxyFlags::empty(),
                };

                // Insert the proxy into the appropriate tree.
                let tree = trees.tree_for_body_mut(*rb);
                let proxy_id = tree.add_proxy(enlarged_aabb, proxy);

                // Store the proxy key.
                *proxy_key = ColliderTreeProxyKey::new(proxy_id, *rb);
            },
        );

        // TODO: Remove proxies when colliders are removed or disabled.
        app.add_observer(
            |trigger: On<Remove, ColliderOf>,
             collider_query: Query<&ColliderTreeProxyKey>,
             mut trees: ResMut<ColliderTrees>| {
                let entity = trigger.entity;

                let Ok(proxy_key) = collider_query.get(entity) else {
                    return;
                };

                let tree = trees.tree_for_body_mut(proxy_key.body());
                tree.remove_proxy(proxy_key.id());
            },
        );
    }
}

/// A resource for tracking proxies whose [`ColliderAabb`] has moved
/// outside of the previous [`EnlargedAabb`].
#[derive(Resource, Default)]
pub struct MovedProxies {
    /// A bit vector tracking moved proxies.
    // TODO: It might be more logical to have these bit vectors in a separate resource.
    bit_vec: MovedProxiesBitVec,
    /// Thread-local bit vectors for tracking moved proxies in parallel.
    ///
    /// These are combined into [`bit_vec`](Self::bit_vec) after parallel processing.
    thread_local_bit_vec: ThreadLocal<RefCell<MovedProxiesBitVec>>,
    /// A vector of moved proxy keys.
    proxies: Vec<ColliderTreeProxyKey>,
    /// A set of moved proxy keys for quick lookup.
    set: HashSet<ColliderTreeProxyKey>,
}

/// Bit vectors for tracking moved dynamic and kinematic proxies.
///
/// Set bits indicate [`ProxyId`]s of moved proxies.
///
/// [`ProxyId`]: crate::collider_tree::ProxyId
#[derive(Default)]
struct MovedProxiesBitVec {
    // Note: Box2D indexes by shape ID, so it only needs one bit vector.
    //       In our case, we would instead index by entity ID, but this would
    //       require a potentially huge and very sparse bit vector since not
    //       all entities are colliders. So we use separate bit vectors for
    //       dynamic and kinematic bodies, and index by proxy ID instead.
    dynamic: BitVec,
    kinematic: BitVec,
}

impl MovedProxies {
    /// Returns the keys of the moved proxies.
    #[inline]
    pub fn proxies(&self) -> &[ColliderTreeProxyKey] {
        &self.proxies
    }

    /// Returns `true` if the proxy with the given key has moved.
    #[inline]
    pub fn contains(&self, proxy_key: ColliderTreeProxyKey) -> bool {
        self.set.contains(&proxy_key)
    }

    /// Clears the moved proxies and sets the capacity of the internal structures.
    #[inline]
    pub fn clear_and_set_capacity(&mut self, dynamic_capacity: usize, kinematic_capacity: usize) {
        self.bit_vec
            .dynamic
            .set_bit_count_and_clear(dynamic_capacity);
        self.bit_vec
            .kinematic
            .set_bit_count_and_clear(kinematic_capacity);

        self.thread_local_bit_vec.iter_mut().for_each(|context| {
            let bit_vec_mut = &mut context.borrow_mut();
            bit_vec_mut
                .dynamic
                .set_bit_count_and_clear(dynamic_capacity);
            bit_vec_mut
                .kinematic
                .set_bit_count_and_clear(kinematic_capacity);
        });

        self.proxies.clear();
        self.set.clear();

        let capacity = dynamic_capacity.max(kinematic_capacity);
        if self.proxies.capacity() < capacity {
            self.proxies.reserve(capacity - self.proxies.capacity());
        }
        if self.set.capacity() < capacity {
            self.set.reserve(capacity - self.set.capacity());
        }
    }

    /// Combines the thread-local moved proxy bit vectors into the main one.
    #[inline]
    pub fn combine_thread_local(&mut self) {
        let bit_vec = &mut self.bit_vec;
        self.thread_local_bit_vec.iter_mut().for_each(|context| {
            let thread_local_bit_vec = context.borrow();
            bit_vec.dynamic.or(&thread_local_bit_vec.dynamic);
            bit_vec.kinematic.or(&thread_local_bit_vec.kinematic);
        });
    }
}

fn update_dynamic_aabbs<C: AnyCollider>(
    mut colliders: ParamSet<(
        Query<(
            Entity,
            &C,
            &mut ColliderAabb,
            &mut EnlargedAabb,
            &ColliderTreeProxyKey,
            &Position,
            &Rotation,
            Option<&CollisionMargin>,
            Option<&SpeculativeMargin>,
        )>,
        Query<(&ColliderAabb, &EnlargedAabb)>,
    )>,
    rb_query: Query<
        (
            &Position,
            &ComputedCenterOfMass,
            &LinearVelocity,
            &AngularVelocity,
            &RigidBodyColliders,
            Has<SweptCcd>,
        ),
        With<SolverBody>,
    >,
    narrow_phase_config: Res<NarrowPhaseConfig>,
    length_unit: Res<PhysicsLengthUnit>,
    mut collider_trees: ResMut<ColliderTrees>,
    mut moved_proxies: ResMut<MovedProxies>,
    time: Res<Time>,
    collider_context: StaticSystemParam<C::Context>,
    mut diagnostics: ResMut<BroadPhaseDiagnostics>,
) {
    let start = crate::utils::Instant::now();

    // An upper bound on the number of dynamic proxies, for sizing the bit vectors.
    // TODO: Use a better way to track the number of proxies.
    let max_num_dynamic_proxies = collider_trees.dynamic_tree.proxies.capacity();
    let max_num_kinematic_proxies = collider_trees.kinematic_tree.proxies.capacity();

    // Clear and resize the moved proxy structures.
    moved_proxies.clear_and_set_capacity(max_num_dynamic_proxies, max_num_kinematic_proxies);

    let delta_secs = time.delta_seconds_adjusted();
    let default_speculative_margin = length_unit.0 * narrow_phase_config.default_speculative_margin;
    let contact_tolerance = length_unit.0 * narrow_phase_config.contact_tolerance;
    let margin = length_unit.0 * 0.05;

    collider_trees
        .dynamic_tree
        .bvh
        .init_primitives_to_nodes_if_uninit();

    let collider_query = colliders.p0();

    rb_query.par_iter().for_each(
        |(rb_pos, center_of_mass, lin_vel, ang_vel, body_colliders, has_swept_ccd)| {
            for collider_entity in body_colliders.iter() {
                let Ok((
                    entity,
                    collider,
                    mut aabb,
                    mut enlarged_aabb,
                    proxy_key,
                    pos,
                    rot,
                    collision_margin,
                    speculative_margin,
                )) = (unsafe { collider_query.get_unchecked(collider_entity) })
                else {
                    continue;
                };

                let collision_margin = collision_margin.map_or(0.0, |margin| margin.0);
                let speculative_margin = if has_swept_ccd {
                    Scalar::MAX
                } else {
                    speculative_margin.map_or(default_speculative_margin, |margin| margin.0)
                };

                let context = AabbContext::new(entity, &*collider_context);

                if speculative_margin <= 0.0 {
                    *aabb = collider
                        .aabb_with_context(pos.0, *rot, context)
                        .grow(Vector::splat(contact_tolerance + collision_margin));
                } else {
                    // If the rigid body is rotating, off-center colliders will orbit around it,
                    // which affects their linear velocities. We need to compute the linear velocity
                    // at the offset position.
                    // TODO: This assumes that the colliders would continue moving in the same direction,
                    //       but because they are orbiting, the direction will change. We should take
                    //       into account the uniform circular motion.
                    let offset = pos.0 - rb_pos.0 - center_of_mass.0;
                    #[cfg(feature = "2d")]
                    let vel = lin_vel.0 + Vector::new(-ang_vel.0 * offset.y, ang_vel.0 * offset.x);
                    #[cfg(feature = "3d")]
                    let vel = lin_vel.0 + ang_vel.cross(offset);
                    let movement = (vel * delta_secs)
                        .clamp_length_max(speculative_margin.max(contact_tolerance));

                    // Current position and predicted position for next feame
                    #[cfg(feature = "2d")]
                    let (end_pos, end_rot) = (
                        pos.0 + movement,
                        *rot * Rotation::radians(ang_vel.0 * delta_secs),
                    );

                    #[cfg(feature = "3d")]
                    let (end_pos, end_rot) = (
                        pos.0 + movement,
                        Rotation(Quaternion::from_scaled_axis(ang_vel.0 * delta_secs) * rot.0)
                            .fast_renormalize(),
                    );

                    // Compute swept AABB, the space that the body would occupy if it was integrated for one frame
                    // TODO: Should we expand the AABB in all directions for speculative contacts?
                    *aabb = collider
                        .swept_aabb_with_context(pos.0, *rot, end_pos, end_rot, context)
                        .grow(Vector::splat(collision_margin));
                }

                let moved = enlarged_aabb.update(&aabb, margin);

                if moved {
                    let mut thread_local_bit_vec = moved_proxies
                        .thread_local_bit_vec
                        .get_or(|| {
                            let mut bit_vec = MovedProxiesBitVec::default();
                            bit_vec
                                .dynamic
                                .set_bit_count_and_clear(max_num_dynamic_proxies);
                            bit_vec
                                .kinematic
                                .set_bit_count_and_clear(max_num_kinematic_proxies);
                            RefCell::new(bit_vec)
                        })
                        .borrow_mut();
                    match proxy_key.body() {
                        RigidBody::Dynamic => {
                            thread_local_bit_vec.dynamic.set(proxy_key.id().index())
                        }
                        RigidBody::Kinematic => {
                            thread_local_bit_vec.kinematic.set(proxy_key.id().index())
                        }
                        RigidBody::Static => {
                            unreachable!(
                                "Static body proxy {proxy_key:?} moved in dynamic AABB update"
                            )
                        }
                    }
                }
            }
        },
    );

    // Combine thread-local moved proxy bit vectors into the main one.
    moved_proxies.combine_thread_local();

    // Serially enlarge moved proxies in the dynamic and kinematic tree.
    let aabbs = colliders.p1();

    let MovedProxies {
        bit_vec,
        proxies: moved_proxies,
        set: moved_set,
        ..
    } = &mut *moved_proxies;

    let ColliderTrees {
        dynamic_tree,
        kinematic_tree,
        ..
    } = &mut *collider_trees;

    dynamic_tree.moved_proxies.clear();
    kinematic_tree.moved_proxies.clear();
    dynamic_tree.bvh.init_primitives_to_nodes_if_uninit();
    kinematic_tree.bvh.init_primitives_to_nodes_if_uninit();

    // TODO: This is kind of ugly, maybe just extract the inner loop into a function?
    for (body_type, tree, bit_vec) in [
        (RigidBody::Dynamic, dynamic_tree, &bit_vec.dynamic),
        (RigidBody::Kinematic, kinematic_tree, &bit_vec.kinematic),
    ] {
        for (i, mut bits) in bit_vec.blocks().enumerate() {
            while bits != 0 {
                let trailing_zeros = bits.trailing_zeros();
                let proxy_id = ProxyId::new(i as u32 * 64 + trailing_zeros);
                let proxy = &mut tree.proxies[proxy_id.index()];
                let entity = proxy.entity;
                let (aabb, enlarged_aabb) = aabbs.get(entity).unwrap_or_else(|_| {
                    panic!(
                        "EnlargedAabb missing for moved collider entity {:?}",
                        entity
                    )
                });

                let aabb = Aabb::from(*aabb);
                let enlarged_aabb = Aabb::from(enlarged_aabb.get());

                // Update the proxy's AABB.
                proxy.aabb = aabb;
                tree.set_proxy_aabb(proxy_id, enlarged_aabb);

                // Record the moved proxy.
                let proxy_key = ColliderTreeProxyKey::new(proxy_id, body_type);
                tree.moved_proxies.push(proxy_id);
                moved_proxies.push(proxy_key);
                moved_set.insert(proxy_key);

                // Clear the least significant set bit
                bits &= bits - 1;
            }
        }

        // Refit the BVH after enlarging proxies.
        // TODO: For a smaller number of moved proxies, it can be faster
        //       to only refit upwards from the moved leaves.
        tree.bvh.refit_all();
    }

    diagnostics.update += start.elapsed();
}

fn update_static_aabbs<C: AnyCollider>(
    static_bodies: Query<&RigidBodyColliders, (Without<SolverBody>, Without<Sleeping>)>,
    mut colliders: Query<
        (
            Entity,
            &Position,
            &Rotation,
            &mut ColliderAabb,
            &C,
            Option<&CollisionMargin>,
            &ColliderTreeProxyKey,
        ),
        Or<(Changed<Position>, Changed<Rotation>, Changed<C>)>,
    >,
    narrow_phase_config: Res<NarrowPhaseConfig>,
    length_unit: Res<PhysicsLengthUnit>,
    mut collider_trees: ResMut<ColliderTrees>,
    mut diagnostics: ResMut<BroadPhaseDiagnostics>,
    collider_context: StaticSystemParam<C::Context>,
) {
    let start = crate::utils::Instant::now();

    let contact_tolerance = length_unit.0 * narrow_phase_config.contact_tolerance;

    collider_trees
        .static_tree
        .bvh
        .init_primitives_to_nodes_if_uninit();

    for body_colliders in &static_bodies {
        let mut iter = colliders.iter_many_mut(body_colliders.iter());
        while let Some((
            entity,
            collider_pos,
            collider_rot,
            mut aabb,
            collider,
            margin,
            proxy_key,
        )) = iter.fetch_next()
        {
            let margin = margin.map_or(0.0, |margin| margin.0);

            let context = AabbContext::new(entity, &*collider_context);

            // Compute the AABB of the collider.
            *aabb = collider
                .aabb_with_context(collider_pos.0, *collider_rot, context)
                .grow(Vector::splat(contact_tolerance + margin));

            // Reinsert the proxy into the BVH.
            collider_trees
                .static_tree
                .reinsert_proxy(proxy_key.id(), Aabb::from(*aabb));
        }
    }

    diagnostics.update += start.elapsed();
}
