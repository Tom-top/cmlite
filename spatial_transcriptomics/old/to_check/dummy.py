########################################################################################################
# Color transformed points
########################################################################################################
#
# volume = REFERENCE.copy() / 5
# volume = volume.astype("uint8")
# volume = 255 - volume
# volume = np.swapaxes(volume, 0, 1)
# volume = np.swapaxes(volume, 1, 2)
# volume_rgb = np.stack([volume, volume, volume], axis=-1)
# # volume_rgba[:, :, :, 3] = 100
# all_classes_dummy = volume_rgb.copy()
#
# # Class
# if unique_cells_class.size != 0:
#     n_classes = len(unique_cells_class[~neuronal_mask_2])
#     print(f"[{m}] {n_classes} unique cell classes found")
#     class_folder = ut.create_dir(os.path.join(SAVING_DIR, "class"))
#     for i, (cls_name, cls_color) in enumerate(zip(unique_cells_class[~neuronal_mask_2],
#                                                   unique_cells_cls_color[~neuronal_mask_2])):
#         class_mask = np.logical_and(np.array(cells_class) == cls_name, ~neuronal_mask)
#         if not np.sum(class_mask) == 0:
#             unique_class_folder = ut.create_dir(os.path.join(class_folder, cls_name))
#             print("\n")
#             print(f"[{m}] Creating volume for class: {cls_name}; {i}/{n_classes}")
#             print(f"[{m}] {np.sum(class_mask)} cells detected")
#             class_points = filtered_points[class_mask]
#
#             # Subclass
#             subcls = np.array(cells_subclass)[class_mask]
#             subcls_c = np.array(cells_subcls_color)[class_mask]
#             subcls_u, idx = np.unique(subcls, return_index=True)
#             subcls_c_u = subcls_c[idx]
#             n_subclasses = len(subcls_u)
#             for j, (subcls_name, subcls_color) in enumerate(zip(subcls_u, subcls_c_u)):
#                 subcls_name = subcls_name.replace("/", "")
#                 subclass_mask = np.logical_and(np.array(cells_subclass) == subcls_name, ~neuronal_mask)
#                 if not np.sum(subclass_mask) == 0:
#                     unique_subclass_folder = ut.create_dir(
#                         os.path.join(unique_class_folder, subcls_name))
#                     print(f"[{m}] Creating volume for subclass: {subcls_name}; {j}/{n_subclasses}")
#                     print(f"[{m}] {np.sum(subclass_mask)} cells detected")
#                     subclass_points = filtered_points[subclass_mask]
#
#                     # Supertype
#                     supertp = np.array(cells_supertype)[subclass_mask]
#                     supertp_c = np.array(cells_supertype_color)[subclass_mask]
#                     supertp_u, idx = np.unique(supertp, return_index=True)
#                     supertp_c_u = supertp_c[idx]
#                     n_supertypes = len(supertp_u)
#                     for k, (supertp_name, supertp_color) in enumerate(zip(supertp_u, supertp_c_u)):
#                         supertp_name = supertp_name.replace("/", "")
#                         supertp_mask = np.logical_and(np.array(cells_supertype) == supertp_name,
#                                                       ~neuronal_mask)
#                         if not np.sum(supertp_mask) == 0:
#                             unique_supertype_folder = ut.create_dir(
#                                 os.path.join(unique_subclass_folder, supertp_name))
#                             supertp_points = filtered_points[supertp_mask]
#
#                             # Cluster
#                             clst = np.array(cells_cluster)[supertp_mask]
#                             clst_c = np.array(cells_cluster_color)[supertp_mask]
#                             clst_u, idx = np.unique(clst, return_index=True)
#                             clst_c_u = clst_c[idx]
#                             n_clusters = len(clst_u)
#                             for l, (clst_name, clst_color) in enumerate(zip(clst_u, clst_c_u)):
#                                 clst_name = clst_name.replace("/", "")
#                                 clst_mask = np.logical_and(np.array(cells_cluster) == clst_name,
#                                                            ~neuronal_mask)
#                                 if not np.sum(clst_mask) == 0:
#                                     unique_cluster_folder = ut.create_dir(
#                                         os.path.join(unique_supertype_folder, clst_name))
#                                     clst_points = filtered_points[clst_mask]
#
#                                     fig = plt.figure()
#                                     ax = plt.subplot(111)
#                                     ax.imshow(np.min(volume_rgb, axis=1))
#                                     ax.scatter(clst_points[:, 2], clst_points[:, 0], c=clst_color, s=1,
#                                                lw=0.1, edgecolors="black", alpha=1)
#                                     st_plt.remove_spines_and_ticks(ax)
#                                     plt.tight_layout()
#                                     plt.savefig(os.path.join(unique_cluster_folder, "max_proj.png"),
#                                                 dpi=500)
#
#                                     # Save the data
#                                     df_points = pd.DataFrame(clst_points, columns=['x', 'y', 'z'])
#                                     df_points['color'] = clst_color
#                                     df_points.to_csv(os.path.join(unique_cluster_folder, 'points.csv'),
#                                                      index=False)
#
#                             fig = plt.figure()
#                             ax = plt.subplot(111)
#                             ax.imshow(np.min(volume_rgb, axis=1))
#                             ax.scatter(supertp_points[:, 2], supertp_points[:, 0], c=supertp_color, s=1,
#                                        lw=0.1, edgecolors="black", alpha=1)
#                             st_plt.remove_spines_and_ticks(ax)
#                             plt.tight_layout()
#                             plt.savefig(os.path.join(unique_supertype_folder, "max_proj.png"), dpi=500)
#
#                             # Save the data
#                             df_points = pd.DataFrame(supertp_points, columns=['x', 'y', 'z'])
#                             df_points['color'] = supertp_color
#                             df_points.to_csv(os.path.join(unique_supertype_folder, 'points.csv'),
#                                              index=False)
#
#                             st_plt.bar_plot(clst_c, clst_u, clst_c_u, unique_supertype_folder)
#
#                     fig = plt.figure()
#                     ax = plt.subplot(111)
#                     ax.imshow(np.min(volume_rgb, axis=1))
#                     ax.scatter(subclass_points[:, 2], subclass_points[:, 0], c=subcls_color, s=1,
#                                lw=0.1, edgecolors="black", alpha=1)
#                     st_plt.remove_spines_and_ticks(ax)
#                     plt.tight_layout()
#                     plt.savefig(os.path.join(unique_subclass_folder, "max_proj.png"), dpi=500)
#
#                     # Save the data
#                     df_points = pd.DataFrame(subclass_points, columns=['x', 'y', 'z'])
#                     df_points['color'] = subcls_color
#                     df_points.to_csv(os.path.join(unique_subclass_folder, 'points.csv'), index=False)
#
#                     st_plt.bar_plot(supertp_c, supertp_u, supertp_c_u, unique_subclass_folder)
#
#             fig = plt.figure()
#             ax = plt.subplot(111)
#             ax.imshow(np.min(volume_rgb, axis=1))
#             ax.scatter(class_points[:, 2], class_points[:, 0], c=cls_color, s=1,
#                        lw=0.1, edgecolors="black", alpha=1)
#             st_plt.remove_spines_and_ticks(ax)
#             plt.tight_layout()
#             plt.savefig(os.path.join(unique_class_folder, "max_proj.png"), dpi=500)
#
#             # Save the data
#             df_points = pd.DataFrame(class_points, columns=['x', 'y', 'z'])
#             df_points['color'] = cls_color
#             df_points.to_csv(os.path.join(unique_class_folder, 'points.csv'), index=False)
#
#             st_plt.bar_plot(subcls_c, subcls_u, subcls_c_u, unique_class_folder)
#
#     st_plt.bar_plot(np.array(cells_cls_color)[~neuronal_mask], unique_cells_class[~neuronal_mask_2],
#                     unique_cells_cls_color[~neuronal_mask_2], class_folder)
