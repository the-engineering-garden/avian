#[cfg(feature = "parallel")]
use core::cell::RefCell;
use core::marker::PhantomData;

use crate::{
    collider_tree::{
        ColliderTreeDiagnostics, ColliderTreeProxy, ColliderTreeProxyKey, ColliderTreeSystems,
        ColliderTreeType, ColliderTrees, ProxyId, tree::ColliderTreeProxyFlags,
    },
    collision::collider::EnlargedAabb,
    data_structures::bit_vec::BitVec,
    dynamics::solver::solver_body::SolverBody,
    prelude::*,
};
use bevy::{
    ecs::{
        change_detection::Tick,
        entity_disabling::Disabled,
        query::QueryFilter,
        system::{StaticSystemParam, SystemChangeTick},
    },
    platform::collections::HashSet,
    prelude::*,
};
use obvhs::aabb::Aabb;
#[cfg(feature = "parallel")]
use thread_local::ThreadLocal;

/// A plugin for updating [`ColliderTree`]s for a collider type `C`.
///
/// [`ColliderTree`]: crate::collider_tree::ColliderTree
pub(super) struct ColliderTreeUpdatePlugin<C: AnyCollider>(PhantomData<C>);

impl<C: AnyCollider> Default for ColliderTreeUpdatePlugin<C> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<C: AnyCollider> Plugin for ColliderTreeUpdatePlugin<C> {
    fn build(&self, app: &mut App) {
        // Initialize resources.
        app.init_resource::<MovedProxies>()
            .init_resource::<EnlargedProxies>()
            .init_resource::<LastDynamicKinematicAabbUpdate>();

        // Add systems for updating collider AABBs before physics step.
        // This accounts for manually moved colliders.
        app.add_systems(
            PhysicsSchedule,
            (
                update_dynamic_kinematic_aabbs::<C>,
                update_static_aabbs::<C>,
                update_standalone_aabbs::<C>,
            )
                .chain()
                .in_set(ColliderTreeSystems::UpdateAabbs)
                // Allowing ambiguities is required so that it's possible
                // to have multiple collision backends at the same time.
                .ambiguous_with_all(),
        );

        // Clear moved proxies and update dynamic and kinematic collider AABBs.
        app.add_systems(
            PhysicsSchedule,
            (clear_moved_proxies, update_dynamic_kinematic_aabbs::<C>)
                .chain()
                .after(PhysicsStepSystems::Finalize)
                .before(PhysicsStepSystems::Last),
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
                &mut EnlargedAabb,
            )>,
             narrow_phase_config: Res<NarrowPhaseConfig>,
             length_unit: Res<PhysicsLengthUnit>,
             collider_context: StaticSystemParam<C::Context>| {
                let contact_tolerance = length_unit.0 * narrow_phase_config.contact_tolerance;
                let aabb_context = AabbContext::new(trigger.entity, &*collider_context);

                if let Ok((collider, pos, rot, collision_margin, mut aabb, mut enlarged_aabb)) =
                    query.get_mut(trigger.entity)
                {
                    // TODO: Should we instead do this in `add_to_tree_on`?
                    let collision_margin = collision_margin.map_or(0.0, |m| m.0);
                    *aabb = collider
                        .aabb_with_context(pos.0, *rot, aabb_context)
                        .grow(Vector::splat(contact_tolerance + collision_margin));
                    enlarged_aabb.update(&aabb, 0.0);
                }
            },
        );

        // Aside from AABB updates, we need to handle the following cases:
        //
        // 1. On insert `C` or `ColliderOf`, add to new tree if not already present. Remove from old tree if present.
        // 2. On remove `C`, remove from tree.
        // 3. On remove `ColliderOf`, move to standalone tree if `C` still exists.
        // 4. On re-enable `C`, add to tree.
        // 5. On disable `C`, remove from tree.
        // 6. On replace `RigidBody`, move attached colliders to new tree.
        // 7. On add `Sensor`, set sensor proxy flag.
        // 8. On remove `Sensor`, unset sensor proxy flag.
        // 9. On replace `CollisionLayers`, update proxy layers.
        // 10. On replace `ActiveCollisionHooks`, set proxy flag.

        // Case 1
        app.add_observer(add_to_tree_on::<Insert, (C, ColliderOf), Without<ColliderDisabled>>);

        // Case 2
        // Note: We also include disabled entities here for the edge case where
        //       we despawn a disabled collider, which causes Case 4 to trigger first.
        //       Ideally Case 4 would not trigger for despawned entities.
        // TODO: Clean up the edge case described above.
        app.add_observer(remove_from_tree_on::<Remove, C, Allow<Disabled>>);

        // Case 3
        app.add_observer(
            |trigger: On<Remove, ColliderOf>,
             mut collider_query: Query<
                (
                    &ColliderTreeProxyKey,
                    &ColliderAabb,
                    &EnlargedAabb,
                    Option<&CollisionLayers>,
                    Has<Sensor>,
                    Option<&ActiveCollisionHooks>,
                ),
                (With<C>, Without<ColliderDisabled>),
            >,
             mut trees: ResMut<ColliderTrees>,
             mut moved_proxies: ResMut<MovedProxies>| {
                let entity = trigger.entity;

                let Ok((proxy_key, collider_aabb, enlarged_aabb, layers, is_sensor, active_hooks)) =
                    collider_query.get_mut(entity)
                else {
                    return;
                };

                // Remove the proxy from its current tree.
                let tree = trees.tree_for_type_mut(proxy_key.tree_type());
                if tree.remove_proxy(proxy_key.id()).is_none() {
                    return;
                }
                moved_proxies.remove(proxy_key);

                // If the collider still exists, move it to the standalone tree.
                let aabb = Aabb::from(*collider_aabb);
                let enlarged_aabb = Aabb::from(enlarged_aabb.get());

                let proxy = ColliderTreeProxy {
                    collider: entity,
                    body: None,
                    layers: layers.copied().unwrap_or_default(),
                    aabb,
                    flags: ColliderTreeProxyFlags::new(
                        is_sensor,
                        active_hooks.copied().unwrap_or_default(),
                    ),
                };

                let standalone_tree = &mut trees.standalone_tree;
                let proxy_id = standalone_tree.add_proxy(enlarged_aabb, proxy);
                let new_proxy_key =
                    ColliderTreeProxyKey::new(proxy_id, ColliderTreeType::Standalone);

                // Mark the proxy as moved.
                moved_proxies.insert(new_proxy_key);
            },
        );

        // Cases 4
        // Note: We use `Replace` here to run before Case 2.
        app.add_observer(
            add_to_tree_on::<Replace, Disabled, (Without<ColliderDisabled>, Allow<Disabled>)>,
        );
        app.add_observer(add_to_tree_on::<Replace, ColliderDisabled, ()>);

        // Case 5
        app.add_observer(
            remove_from_tree_on::<Add, Disabled, (Without<ColliderDisabled>, Allow<Disabled>)>,
        );
        app.add_observer(remove_from_tree_on::<Add, ColliderDisabled, ()>);

        // Case 6
        app.add_observer(
            |trigger: On<Insert, RigidBody>,
             body_query: Query<(&RigidBody, &RigidBodyColliders)>,
             mut collider_query: Query<
                (
                    &ColliderAabb,
                    &EnlargedAabb,
                    &mut ColliderTreeProxyKey,
                    Option<&CollisionLayers>,
                    Has<Sensor>,
                    Option<&ActiveCollisionHooks>,
                ),
                Without<ColliderDisabled>,
            >,
             mut trees: ResMut<ColliderTrees>,
             mut moved_proxies: ResMut<MovedProxies>| {
                let entity = trigger.entity;

                let Ok((new_rb, body_colliders)) = body_query.get(entity) else {
                    return;
                };

                for collider_entity in body_colliders.iter() {
                    let Ok((
                        collider_aabb,
                        enlarged_aabb,
                        mut proxy_key,
                        layers,
                        is_sensor,
                        active_hooks,
                    )) = collider_query.get_mut(collider_entity)
                    else {
                        continue;
                    };

                    let new_tree_type = ColliderTreeType::from_body(Some(*new_rb));

                    if new_tree_type == proxy_key.tree_type() {
                        // No tree change.
                        break;
                    }

                    // Remove the old proxy from its current tree.
                    let old_tree = trees.tree_for_type_mut(proxy_key.tree_type());
                    old_tree.remove_proxy(proxy_key.id());
                    moved_proxies.remove(&proxy_key);

                    // Insert the proxy into the new tree.
                    let aabb = Aabb::from(*collider_aabb);
                    let enlarged_aabb = Aabb::from(enlarged_aabb.get());

                    let proxy = ColliderTreeProxy {
                        collider: collider_entity,
                        body: Some(entity),
                        layers: layers.copied().unwrap_or_default(),
                        aabb,
                        flags: ColliderTreeProxyFlags::new(
                            is_sensor,
                            active_hooks.copied().unwrap_or_default(),
                        ),
                    };

                    let new_tree = trees.tree_for_type_mut(new_tree_type);
                    let proxy_id = new_tree.add_proxy(enlarged_aabb, proxy);
                    let new_proxy_key = ColliderTreeProxyKey::new(proxy_id, new_tree_type);

                    // Store the new proxy key.
                    *proxy_key = new_proxy_key;

                    // Mark the proxy as moved.
                    moved_proxies.insert(new_proxy_key);
                }
            },
        );

        // Case 7
        app.add_observer(
            |trigger: On<Add, Sensor>,
             mut collider_query: Query<&ColliderTreeProxyKey, Without<ColliderDisabled>>,
             mut trees: ResMut<ColliderTrees>| {
                let entity = trigger.entity;

                let Ok(proxy_key) = collider_query.get_mut(entity) else {
                    return;
                };

                let tree = trees.tree_for_type_mut(proxy_key.tree_type());

                // Set sensor flag.
                if let Some(proxy) = tree.get_proxy_mut(proxy_key.id()) {
                    proxy.flags.insert(ColliderTreeProxyFlags::SENSOR);
                }
            },
        );

        // Case 8
        app.add_observer(
            |trigger: On<Remove, Sensor>,
             mut collider_query: Query<&ColliderTreeProxyKey, Without<ColliderDisabled>>,
             mut trees: ResMut<ColliderTrees>| {
                let entity = trigger.entity;

                let Ok(proxy_key) = collider_query.get_mut(entity) else {
                    return;
                };

                let tree = trees.tree_for_type_mut(proxy_key.tree_type());

                // Unset sensor flag.
                if let Some(proxy) = tree.get_proxy_mut(proxy_key.id()) {
                    proxy.flags.remove(ColliderTreeProxyFlags::SENSOR);
                }
            },
        );

        // Case 9
        app.add_observer(
            |trigger: On<Replace, CollisionLayers>,
             mut collider_query: Query<
                (&ColliderTreeProxyKey, Option<&CollisionLayers>),
                Without<ColliderDisabled>,
            >,
             mut trees: ResMut<ColliderTrees>| {
                let entity = trigger.entity;

                let Ok((proxy_key, layers)) = collider_query.get_mut(entity) else {
                    return;
                };

                let tree = trees.tree_for_type_mut(proxy_key.tree_type());

                // Update layers.
                if let Some(proxy) = tree.get_proxy_mut(proxy_key.id()) {
                    proxy.layers = layers.copied().unwrap_or_default();
                }
            },
        );

        // Case 10
        app.add_observer(
            |trigger: On<Replace, ActiveCollisionHooks>,
             mut collider_query: Query<
                (&ColliderTreeProxyKey, Option<&ActiveCollisionHooks>),
                Without<ColliderDisabled>,
            >,
             mut trees: ResMut<ColliderTrees>| {
                let entity = trigger.entity;

                let Ok((proxy_key, active_hooks)) = collider_query.get_mut(entity) else {
                    return;
                };

                let tree = trees.tree_for_type_mut(proxy_key.tree_type());

                // Update active hooks flags.
                if let Some(proxy) = tree.get_proxy_mut(proxy_key.id()) {
                    proxy.flags.set(
                        ColliderTreeProxyFlags::CUSTOM_FILTER,
                        active_hooks
                            .is_some_and(|h| h.contains(ActiveCollisionHooks::FILTER_PAIRS)),
                    );
                }
            },
        );
    }
}

/// Adds a collider to the appropriate collider tree when the event `E` is triggered.
fn add_to_tree_on<E: EntityEvent, B: Bundle, F: QueryFilter>(
    trigger: On<E, B>,
    body_query: Query<&RigidBody, Allow<Disabled>>,
    mut collider_query: Query<
        (
            Option<&ColliderOf>,
            &ColliderAabb,
            &EnlargedAabb,
            &mut ColliderTreeProxyKey,
            Option<&CollisionLayers>,
            Has<Sensor>,
            Option<&ActiveCollisionHooks>,
        ),
        F,
    >,
    mut trees: ResMut<ColliderTrees>,
    mut moved_proxies: ResMut<MovedProxies>,
) {
    let entity = trigger.event_target();

    let Ok((
        collider_of,
        collider_aabb,
        enlarged_aabb,
        mut proxy_key,
        layers,
        is_sensor,
        active_hooks,
    )) = collider_query.get_mut(entity)
    else {
        return;
    };

    let tree_type = if let Some(Ok(rb)) = collider_of.map(|c| body_query.get(c.body)) {
        ColliderTreeType::from_body(Some(*rb))
    } else {
        ColliderTreeType::Standalone
    };

    let aabb = Aabb::from(*collider_aabb);
    let enlarged_aabb = Aabb::from(enlarged_aabb.get());

    let proxy = ColliderTreeProxy {
        collider: entity,
        body: collider_of.map(|c| c.body),
        layers: layers.copied().unwrap_or_default(),
        aabb,
        flags: ColliderTreeProxyFlags::new(is_sensor, active_hooks.copied().unwrap_or_default()),
    };

    // Remove the old proxy if it exists.
    if *proxy_key != ColliderTreeProxyKey::PLACEHOLDER {
        let old_tree_type = proxy_key.tree_type();
        let old_tree = trees.tree_for_type_mut(old_tree_type);
        old_tree.remove_proxy(proxy_key.id());
        moved_proxies.remove(&proxy_key);
    }

    // Insert the proxy into the appropriate tree.
    let tree = trees.tree_for_type_mut(tree_type);
    let proxy_id = tree.add_proxy(enlarged_aabb, proxy);

    // Store the proxy key.
    *proxy_key = ColliderTreeProxyKey::new(proxy_id, tree_type);

    // Mark the proxy as moved.
    moved_proxies.insert(*proxy_key);
}

/// Removes a collider from its collider tree when the event `E` is triggered.
fn remove_from_tree_on<E: EntityEvent, B: Bundle, F: QueryFilter>(
    trigger: On<E, B>,
    mut collider_query: Query<&mut ColliderTreeProxyKey, F>,
    mut trees: ResMut<ColliderTrees>,
    mut moved_proxies: ResMut<MovedProxies>,
) {
    let entity = trigger.event_target();

    let Ok(mut proxy_key) = collider_query.get_mut(entity) else {
        return;
    };

    if *proxy_key == ColliderTreeProxyKey::PLACEHOLDER {
        return;
    }

    // Remove the proxy from its current tree.
    let tree = trees.tree_for_type_mut(proxy_key.tree_type());
    tree.remove_proxy(proxy_key.id());
    moved_proxies.remove(&proxy_key);

    // Invalidate the proxy key.
    *proxy_key = ColliderTreeProxyKey::PLACEHOLDER;
}

/// A resource for tracking the last system change tick
/// when dynamic or kinematic collider AABBs were updated.
#[derive(Resource, Default)]
struct LastDynamicKinematicAabbUpdate(Tick);

/// A resource for tracking moved proxies.
///
/// Moved proxies are those whose [`ColliderAabb`] has moved outside of their
/// previous [`EnlargedAabb`], or whose collider has been added to a [`ColliderTree`].
///
/// [`ColliderTree`]: crate::collider_tree::ColliderTree
#[derive(Resource, Default)]
pub struct MovedProxies {
    /// A vector of moved proxy keys.
    proxies: Vec<ColliderTreeProxyKey>,
    /// A set of moved proxy keys for quick lookup.
    set: HashSet<ColliderTreeProxyKey>,
}

impl MovedProxies {
    /// Returns the keys of the moved proxies.
    ///
    /// The order of the keys is the order in which they were inserted.
    #[inline]
    pub fn proxies(&self) -> &[ColliderTreeProxyKey] {
        &self.proxies
    }

    /// Returns `true` if the proxy with the given key has moved.
    #[inline]
    pub fn contains(&self, proxy_key: ColliderTreeProxyKey) -> bool {
        self.set.contains(&proxy_key)
    }

    /// Inserts a moved proxy key.
    ///
    /// If the proxy key is already present, it is not added again.
    #[inline]
    pub fn insert(&mut self, proxy_key: ColliderTreeProxyKey) {
        if self.set.insert(proxy_key) {
            self.proxies.push(proxy_key);
        }
    }

    /// Removes a moved proxy key. This uses a linear search,
    /// and may change the order of the remaining keys.
    ///
    /// If the proxy key is not present, nothing happens.
    #[inline]
    pub fn remove(&mut self, proxy_key: &ColliderTreeProxyKey) {
        if self.set.remove(proxy_key)
            && let Some(pos) = self.proxies.iter().position(|k| k == proxy_key)
        {
            self.proxies.swap_remove(pos);
        }
    }

    /// Clears the moved proxies.
    #[inline]
    pub fn clear(&mut self) {
        self.proxies.clear();
        self.set.clear();
    }
}

/// Bit vectors for tracking dynamic and kinematic proxies whose
/// [`ColliderAabb`] has moved outside of the previous [`EnlargedAabb`].
///
/// Set bits indicate [`ProxyId`]s of moved proxies.
#[derive(Resource, Default)]
struct EnlargedProxies {
    bit_vec: EnlargedProxiesBitVec,
    thread_local_bit_vec: ThreadLocal<RefCell<EnlargedProxiesBitVec>>,
}

/// Bit vectors for tracking moved dynamic and kinematic proxies.
///
/// Set bits indicate [`ProxyId`]s of moved proxies.
///
/// [`ProxyId`]: crate::collider_tree::ProxyId
#[derive(Default)]
struct EnlargedProxiesBitVec {
    // Note: Box2D indexes by shape ID, so it only needs one bit vector.
    //       In our case, we would instead index by entity ID, but this would
    //       require a potentially huge and very sparse bit vector since not
    //       all entities are colliders. So we use separate bit vectors for
    //       dynamic and kinematic bodies, and index by proxy ID instead.
    dynamic: BitVec,
    kinematic: BitVec,
}

impl EnlargedProxies {
    /// Clears the enlarged proxies and sets the capacity of the internal structures.
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
    }

    /// Combines the thread-local enlarged proxy bit vectors into the main one.
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

/// Updates the AABBs of colliders attached to dynamic or kinematic rigid bodies.
// TODO: Optimize the change detection.
fn update_dynamic_kinematic_aabbs<C: AnyCollider>(
    mut colliders: ParamSet<(
        Query<
            (
                Ref<C>,
                &mut ColliderAabb,
                &mut EnlargedAabb,
                &ColliderTreeProxyKey,
                Ref<Position>,
                Ref<Rotation>,
                Option<&CollisionMargin>,
                Option<&SpeculativeMargin>,
            ),
            Without<ColliderDisabled>,
        >,
        Query<(&ColliderAabb, &EnlargedAabb), Without<ColliderDisabled>>,
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
    mut enlarged_proxies: ResMut<EnlargedProxies>,
    time: Res<Time>,
    collider_context: StaticSystemParam<C::Context>,
    mut diagnostics: ResMut<ColliderTreeDiagnostics>,
    mut last_tick: ResMut<LastDynamicKinematicAabbUpdate>,
    system_tick: SystemChangeTick,
) {
    let start = crate::utils::Instant::now();

    let this_run = system_tick.this_run();

    // An upper bound on the number of proxies, for sizing the bit vectors.
    // TODO: Use a better way to track the number of proxies.
    let max_num_dynamic_proxies = collider_trees.dynamic_tree.proxies.capacity();
    let max_num_kinematic_proxies = collider_trees.kinematic_tree.proxies.capacity();

    // Clear and resize the enlarged proxy structures.
    enlarged_proxies.clear_and_set_capacity(max_num_dynamic_proxies, max_num_kinematic_proxies);

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

                // Skip if the collider's AABB can't have changed since the last physics tick.
                if !pos.last_changed().is_newer_than(last_tick.0, this_run)
                    && !rot.last_changed().is_newer_than(last_tick.0, this_run)
                    && !collider.last_changed().is_newer_than(last_tick.0, this_run)
                {
                    continue;
                }

                let collision_margin = collision_margin.map_or(0.0, |margin| margin.0);
                let speculative_margin = if has_swept_ccd {
                    Scalar::MAX
                } else {
                    speculative_margin.map_or(default_speculative_margin, |margin| margin.0)
                };

                let context = AabbContext::new(collider_entity, &*collider_context);

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
                    let mut thread_local_bit_vec = enlarged_proxies
                        .thread_local_bit_vec
                        .get_or(|| {
                            let mut bit_vec = EnlargedProxiesBitVec::default();
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
                        Some(RigidBody::Dynamic) => {
                            thread_local_bit_vec.dynamic.set(proxy_key.id().index())
                        }
                        Some(RigidBody::Kinematic) => {
                            thread_local_bit_vec.kinematic.set(proxy_key.id().index())
                        }
                        _ => {
                            unreachable!("Static proxy {proxy_key:?} moved in dynamic AABB update")
                        }
                    }
                }
            }
        },
    );

    // Combine thread-local moved proxy bit vectors into the main one.
    enlarged_proxies.combine_thread_local();

    // Serially enlarge moved proxies in the dynamic and kinematic tree.
    let aabbs = colliders.p1();

    let ColliderTrees {
        dynamic_tree,
        kinematic_tree,
        ..
    } = &mut *collider_trees;

    dynamic_tree.bvh.init_primitives_to_nodes_if_uninit();
    kinematic_tree.bvh.init_primitives_to_nodes_if_uninit();

    // TODO: This is kind of ugly, maybe just extract the inner loop into a function?
    for (tree_type, tree, bit_vec) in [
        (
            ColliderTreeType::Dynamic,
            dynamic_tree,
            &enlarged_proxies.bit_vec.dynamic,
        ),
        (
            ColliderTreeType::Kinematic,
            kinematic_tree,
            &enlarged_proxies.bit_vec.kinematic,
        ),
    ] {
        for (i, mut bits) in bit_vec.blocks().enumerate() {
            while bits != 0 {
                let trailing_zeros = bits.trailing_zeros();
                let proxy_id = ProxyId::new(i as u32 * 64 + trailing_zeros);
                let proxy = &mut tree.proxies[proxy_id.index()];
                let entity = proxy.collider;
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
                let proxy_key = ColliderTreeProxyKey::new(proxy_id, tree_type);
                tree.moved_proxies.push(proxy_id);
                moved_proxies.insert(proxy_key);

                // Clear the least significant set bit
                bits &= bits - 1;
            }
        }

        // Refit the BVH after enlarging proxies.
        // TODO: For a smaller number of moved proxies, it can be faster
        //       to only refit upwards from the moved leaves.
        tree.bvh.refit_all();
    }

    // Update the last update tick.
    last_tick.0 = this_run;

    diagnostics.update += start.elapsed();
}

// TODO: If we tagged static colliders with their own marker component,
//       we could avoid querying for the bodies, and merge this with the
//       standalone collider AABB update.
/// Updates the AABBs of colliders attached to static rigid bodies.
fn update_static_aabbs<C: AnyCollider>(
    static_bodies: Query<&RigidBodyColliders, (Without<SolverBody>, Without<Sleeping>)>,
    mut colliders: Query<
        (
            Entity,
            &Position,
            &Rotation,
            &mut ColliderAabb,
            &mut EnlargedAabb,
            &C,
            Option<&CollisionMargin>,
            &ColliderTreeProxyKey,
        ),
        (
            Without<ColliderDisabled>,
            Or<(Changed<Position>, Changed<Rotation>, Changed<C>)>,
        ),
    >,
    narrow_phase_config: Res<NarrowPhaseConfig>,
    length_unit: Res<PhysicsLengthUnit>,
    mut collider_trees: ResMut<ColliderTrees>,
    mut diagnostics: ResMut<ColliderTreeDiagnostics>,
    collider_context: StaticSystemParam<C::Context>,
) {
    let start = crate::utils::Instant::now();

    let contact_tolerance = length_unit.0 * narrow_phase_config.contact_tolerance;

    collider_trees
        .static_tree
        .bvh
        .init_primitives_to_nodes_if_uninit();

    // TODO: Parallelize this and/or avoid iterating over all static bodies.
    // TODO: Enlarged AABBs are not really needed for static colliders.
    for body_colliders in &static_bodies {
        let mut iter = colliders.iter_many_mut(body_colliders.iter());
        while let Some((
            entity,
            collider_pos,
            collider_rot,
            mut aabb,
            mut enlarged_aabb,
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
            enlarged_aabb.update(&aabb, 0.0);

            // Reinsert the proxy into the BVH.
            collider_trees
                .static_tree
                .reinsert_proxy(proxy_key.id(), Aabb::from(*aabb));
        }
    }

    diagnostics.update += start.elapsed();
}

/// Updates the AABBs of standalone colliders that are not attached to any rigid body.
fn update_standalone_aabbs<C: AnyCollider>(
    mut colliders: Query<
        (
            Entity,
            &Position,
            &Rotation,
            &mut ColliderAabb,
            &mut EnlargedAabb,
            &C,
            Option<&CollisionMargin>,
            &ColliderTreeProxyKey,
        ),
        (
            Without<ColliderOf>,
            Without<ColliderDisabled>,
            Or<(Changed<Position>, Changed<Rotation>, Changed<C>)>,
        ),
    >,
    narrow_phase_config: Res<NarrowPhaseConfig>,
    length_unit: Res<PhysicsLengthUnit>,
    mut collider_trees: ResMut<ColliderTrees>,
    mut diagnostics: ResMut<ColliderTreeDiagnostics>,
    collider_context: StaticSystemParam<C::Context>,
) {
    let start = crate::utils::Instant::now();

    let contact_tolerance = length_unit.0 * narrow_phase_config.contact_tolerance;

    collider_trees
        .standalone_tree
        .bvh
        .init_primitives_to_nodes_if_uninit();

    for (
        entity,
        collider_pos,
        collider_rot,
        mut aabb,
        mut enlarged_aabb,
        collider,
        margin,
        proxy_key,
    ) in &mut colliders
    {
        let margin = margin.map_or(0.0, |margin| margin.0);

        let context = AabbContext::new(entity, &*collider_context);

        // Compute the AABB of the collider.
        *aabb = collider
            .aabb_with_context(collider_pos.0, *collider_rot, context)
            .grow(Vector::splat(contact_tolerance + margin));
        enlarged_aabb.update(&aabb, 0.0);

        // Reinsert the proxy into the BVH.
        collider_trees
            .standalone_tree
            .reinsert_proxy(proxy_key.id(), Aabb::from(*aabb));
    }

    diagnostics.update += start.elapsed();
}

fn clear_moved_proxies(
    mut moved_proxies: ResMut<MovedProxies>,
    mut collider_trees: ResMut<ColliderTrees>,
) {
    moved_proxies.clear();
    collider_trees.dynamic_tree.moved_proxies.clear();
    collider_trees.kinematic_tree.moved_proxies.clear();
    collider_trees.static_tree.moved_proxies.clear();
    collider_trees.standalone_tree.moved_proxies.clear();
}
