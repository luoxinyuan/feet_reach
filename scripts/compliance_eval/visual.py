"""Streaming 540p H.264 video, reference mesh ghost and actual force arrows."""
import numpy as np
class Video:
    def __init__(self,env,cmd,path,tracking,prompt_only=False):
        import omni.usd
        from pxr import Usd,UsdGeom,UsdShade,Sdf,Gf
        import imageio.v2 as imageio
        from PIL import ImageFont
        env.base_env.lookat_env_i=0
        self.prompt_only=prompt_only
        self.env,self.cmd,self.tracking=env,cmd,tracking;self.Gf=Gf;self.UsdGeom=UsdGeom
        self.writer=imageio.get_writer(path,fps=25,codec='libx264',macro_block_size=1,ffmpeg_params=['-crf','28','-maxrate','1200k','-bufsize','2400k','-pix_fmt','yuv420p'])
        self.font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
        stage=omni.usd.get_context().get_stage();cache=UsdGeom.XformCache()
        def material(path,color,opacity=1.):
            mat=UsdShade.Material.Define(stage,path);shader=UsdShade.Shader.Define(stage,path+'/Shader');shader.CreateIdAttr('UsdPreviewSurface')
            shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color));shader.CreateInput('opacity',Sdf.ValueTypeNames.Float).Set(opacity)
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface');return mat
        self.ghost=[]
        if tracking:
            cyan=material('/World/Eval/GhostMaterial',(.05,.85,1.),.45 if prompt_only else .2)
            root=stage.GetPrimAtPath('/World/envs/env_0/Robot');bodies={pr.GetName():pr for pr in Usd.PrimRange(root) if pr.GetName() in cmd.asset.body_names}
            count=0
            for name,body in bodies.items():
                if name not in cmd.dataset.body_names:continue
                xf=UsdGeom.Xform.Define(stage,'/World/Eval/Ghost/'+name);self.ghost.append((xf.AddTransformOp(),cmd.dataset.body_names.index(name)));bw=cache.GetLocalToWorldTransform(body)
                for j,prim in enumerate(Usd.PrimRange(body,Usd.TraverseInstanceProxies())):
                    if not prim.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(prim).ComputeVisibility()=='invisible':continue
                    src=UsdGeom.Mesh(prim);dst=UsdGeom.Mesh.Define(stage,str(xf.GetPath())+'/mesh'+str(j));count+=1
                    for key in ['points','faceVertexCounts','faceVertexIndices','normals','subdivisionScheme','orientation','doubleSided']:
                        attr=prim.GetAttribute(key)
                        if attr and attr.HasAuthoredValueOpinion():dst.GetPrim().CreateAttribute(key,attr.GetTypeName()).Set(attr.Get())
                    dst.SetNormalsInterpolation(src.GetNormalsInterpolation());UsdGeom.Xformable(dst).AddTransformOp().Set(cache.GetLocalToWorldTransform(prim)*bw.GetInverse());UsdShade.MaterialBindingAPI.Apply(dst.GetPrim()).Bind(cyan)
            assert count>10
        self.arrows=[]
        for i,color in enumerate([(1.,.3,.05),(1.,.75,.05),(.95,.15,.65)]):
            mat=material(f'/World/Eval/FMaterial{i}',color)
            shaft=UsdGeom.Cylinder.Define(stage,f'/World/Eval/F{i}/shaft');shaft.CreateAxisAttr('Z');shaft.CreateRadiusAttr(.012)
            head=UsdGeom.Cone.Define(stage,f'/World/Eval/F{i}/head');head.CreateAxisAttr('Z');head.CreateRadiusAttr(.035)
            for shape in (shaft,head):UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(mat)
            self.arrows.append((shaft,shaft.AddTransformOp(),head,head.AddTransformOp()))
    def matrix(self,p,q):
        Gf=self.Gf;m=Gf.Matrix4d(1.);m.SetRotate(Gf.Quatd(float(q[0]),Gf.Vec3d(*map(float,q[1:]))));m.SetTranslateOnly(Gf.Vec3d(*map(float,p)));return m
    def frame(self,t,label,fallen):
        from PIL import Image,ImageDraw
        cmd=self.cmd;robot=cmd.asset;Gf=self.Gf
        if self.tracking:
            pos=cmd._motion.body_pos_w[0,0].cpu().numpy()+self.env.scene.env_origins[0].cpu().numpy();quat=cmd._motion.body_quat_w[0,0].cpu().numpy()
            for op,i in self.ghost:op.Set(self.matrix(pos[i],quat[i]))
        force=cmd.force_apply_buffer[0].cpu().numpy();pos=cmd.position_apply_buffer[0].cpu().numpy();names=robot.body_names
        if self.prompt_only:
            import torch
            from active_adaptation.utils.math import quat_apply
            points=cmd.cache_manifest['points'];bids=[robot.body_names.index(p['body']) for p in points]
            offsets=torch.tensor([p['offset'] for p in points],device=robot.device,dtype=torch.float32)
            pos=(robot.data.body_pos_w[0,bids]+quat_apply(robot.data.body_quat_w[0,bids],offsets)).cpu().numpy()
            force=cmd.point_force_buffer[0].cpu().numpy();names=[p['name'] for p in points]
        ids=np.flatnonzero(np.linalg.norm(force,axis=-1)>.05)
        if self.prompt_only:ids=ids[np.argsort(-np.linalg.norm(force[ids],axis=-1))][:3]
        for i,(shaft,sop,head,hop) in enumerate(self.arrows):
            visible=i<len(ids)
            for shape in (shaft,head):self.UsdGeom.Imageable(shape).GetVisibilityAttr().Set('inherited' if visible else 'invisible')
            if not visible:continue
            idx=ids[i];mag=np.linalg.norm(force[idx]);direction=force[idx]/mag;length=min(1.2,.045*mag);tip=min(.08,.4*length)
            q=Gf.Rotation(Gf.Vec3d(0,0,1),Gf.Vec3d(*map(float,direction))).GetQuat();quat=[q.GetReal(),*q.GetImaginary()]
            shaft.CreateHeightAttr(max(.001,length-tip));head.CreateHeightAttr(tip);sop.Set(self.matrix(pos[idx]+direction*(length-tip)/2,quat));hop.Set(self.matrix(pos[idx]+direction*(length-tip/2),quat))
        root=robot.data.root_pos_w[0].cpu().numpy();center=root.copy();center[2]=.7
        self.env.base_env.cfg.viewer.eye=(center+np.array([2.5,-3.,1.4])-root).tolist();self.env.base_env.cfg.viewer.lookat=(center-root).tolist()
        image=Image.fromarray(self.env.render('rgb_array'));draw=ImageDraw.Draw(image,'RGBA')
        if self.prompt_only and self.tracking:
            import omni.usd
            from pxr import Usd
            stage=omni.usd.get_context().get_stage()
            cam=self.UsdGeom.Camera(stage.GetPrimAtPath('/OmniverseKit_Persp')).GetCamera(Usd.TimeCode.Default())
            vp=cam.frustum.ComputeViewMatrix()*cam.frustum.ComputeProjectionMatrix()
            target=cmd._motion.body_pos_w[0,0].cpu().numpy()+self.env.scene.env_origins[0].cpu().numpy()
            def project(p):
                v=Gf.Vec4d(*map(float,p),1.)*vp
                return ((float(v[0]/v[3])*.5+.5)*image.width,(.5-float(v[1]/v[3])*.5)*image.height)
            chains=[['pelvis','torso_link','head_mimic']]
            for side in ['left','right']:
                chains += [['torso_link',side+'_shoulder_roll_link',side+'_elbow_link',side+'_wrist_roll_link',side+'_hand_mimic'],['pelvis',side+'_hip_roll_link',side+'_knee_link',side+'_ankle_roll_link']]
            for chain in chains:
                xy=[project(target[cmd.dataset.body_names.index(name)]) for name in chain if name in cmd.dataset.body_names]
                if len(xy)>1:draw.line(xy,fill=(40,230,255,200),width=3)
        draw.rectangle((0,0,960,78),fill=(10,16,25,225))
        phase='REFERENCE TRACKING' if self.tracking else 'NO FORCE' if t<2 else 'FORCE' if t<8 else 'RELEASED'
        draw.text((12,8),f'{label} | {t:.2f} s | {phase}'+(' | FALL' if fallen else ''),font=self.font,fill='white')
        draw.text((12,32),'cyan = target | top 3 prompt arrows | physical force = 0 N' if self.prompt_only else 'cyan = target robot | arrows = actual force' if self.tracking else 'No motion target | arrows = physical applied forces',font=self.font,fill=(100,225,255))
        draw.text((12,54),' | '.join(f'{names[i]} {np.linalg.norm(force[i]):.1f} N' for i in ids),font=self.font,fill=(255,210,110))
        self.writer.append_data(np.asarray(image))
    def close(self):self.writer.close()
