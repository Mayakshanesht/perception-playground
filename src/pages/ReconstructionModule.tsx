import ModulePage from "@/components/ModulePage";
import { ModuleContent } from "@/data/moduleContent";
import { moduleContents } from "@/data/moduleContent";

// Merge SfM + NeRF + add MVS into 3D reconstruction
const mvsTheory = [
  {
    title: "Multi-View Stereo (MVS)",
    content: "After SfM produces calibrated cameras and a sparse point cloud, Multi-View Stereo (MVS) computes dense geometry. MVS estimates depth maps from multiple calibrated views, performs multi-view consistency checks, and fuses per-view depth maps into a dense 3D representation (point cloud, mesh, or voxel grid). Methods range from traditional plane sweeping and PatchMatch to learned approaches like MVSNet.",
    equations: [
      {
        label: "Photo-Consistency Cost",
        tex: "C(\\mathbf{p}, d) = \\frac{1}{N-1} \\sum_{i=2}^{N} \\rho(I_1(\\mathbf{p}), I_i(\\mathcal{H}_i(\\mathbf{p}, d)))",
      },
      {
        label: "Depth Map Fusion",
        tex: "D_{\\text{fused}}(\\mathbf{p}) = \\text{median}\\{D_i(\\pi_i(\\mathbf{X}_\\mathbf{p}))\\}_{i=1}^{N}",
      },
    ],
  },
];

const reconstructionModule: ModuleContent = {
  id: "reconstruction",
  title: "3D Reconstruction & Rendering",
  subtitle: "Reconstruct 3D scenes from images and render novel views — from classical Structure from Motion through Multi-View Stereo to neural radiance fields and Gaussian Splatting.",
  color: "340 75% 55%",
  theory: [
    ...moduleContents.sfm.theory,
    ...mvsTheory,
    ...moduleContents.nerf.theory,
  ],
  algorithms: [
    ...moduleContents.sfm.algorithms,
    {
      name: "MVS Pipeline (COLMAP Dense)",
      steps: [
        { step: "Input from SfM", detail: "Calibrated cameras, sparse 3D points from incremental SfM" },
        { step: "Stereo Matching", detail: "PatchMatch stereo computes depth and normal maps for each view" },
        { step: "Consistency Filtering", detail: "Check geometric and photometric consistency across views" },
        { step: "Depth Map Fusion", detail: "Merge per-view depth maps into a unified dense point cloud" },
        { step: "Surface Reconstruction", detail: "Poisson or Delaunay meshing from dense point cloud" },
      ],
    },
    ...moduleContents.nerf.algorithms,
  ],
  papers: [
    ...moduleContents.sfm.papers,
    { year: 2010, title: "PMVS2", authors: "Furukawa & Ponce", venue: "TPAMI", summary: "Patch-based Multi-View Stereo for dense reconstruction with expansion and filtering." },
    { year: 2018, title: "MVSNet", authors: "Yao et al.", venue: "ECCV", summary: "Learned multi-view stereo with differentiable homography warping and 3D cost volume." },
    ...moduleContents.nerf.papers,
  ].sort((a, b) => a.year - b.year),
};

export default function ReconstructionModule() {
  return <ModulePage content={reconstructionModule} />;
}
