# Separate on- and off-screen sound from video file. See README for usage examples.

import aolib.util as ut, aolib.img as ig, os, numpy as np, tensorflow as tf, tfutil as mu, scipy.io, sys, aolib.imtable as imtable, pylab, argparse, shift_params, shift_net
import sourcesep, sep_params
import aolib.sound as sound
from aolib.sound import Sound
import cv2

import multiprocessing
from multiprocessing import Process
from multiprocessing import Pool
import copy
import traceback
from datetime import date

def get_current_time():
  today= date.today()
  t = today.strftime("%d-%m-%Y_%H:%M:%S")
  return t

pj = ut.pjoin
print "Begining of trhe program"

# De vu add log
def write_log(log):
    with open("error.txt", "a") as f:
        f.write(log)
        f.write("\n")

class NetClf:
  def __init__(self, pr, sess = None, gpu = None, restore_only_shift = False):
    self.pr = pr
    self.sess = sess
    self.gpu = gpu
    self.restore_only_shift = restore_only_shift

  def init(self, reset = True):
    if self.sess is None:
      print 'Running on:', self.gpu
      with tf.device(self.gpu):
        if reset:
          tf.reset_default_graph()
          tf.Graph().as_default()
        pr = self.pr
        self.sess = tf.Session()
        self.ims_ph = tf.placeholder(
          tf.uint8, [1, pr.sampled_frames, pr.crop_im_dim, pr.crop_im_dim, 3])
        self.samples_ph = tf.placeholder(tf.float32, (1, pr.num_samples, 2))
        
        crop_spec = lambda x : x[:, :pr.spec_len]
        samples_trunc = self.samples_ph[:, :pr.sample_len]

        spec_mix, phase_mix = sourcesep.stft(samples_trunc[:, :, 0], pr)
        print 'Raw spec length:', mu.shape(spec_mix)
        spec_mix = crop_spec(spec_mix)
        phase_mix = crop_spec(phase_mix)
        print 'Truncated spec length:', mu.shape(spec_mix)

        self.specgram_op, phase = map(crop_spec, sourcesep.stft(samples_trunc[:, :, 0], pr))
        self.auto_op = sourcesep.istft(self.specgram_op, phase, pr)

        self.net = sourcesep.make_net(
          self.ims_ph, samples_trunc, spec_mix, phase_mix, 
          pr, reuse = False, train = False)
        self.spec_pred_fg = self.net.pred_spec_fg
        self.spec_pred_bg = self.net.pred_spec_bg
        self.samples_pred_fg = self.net.pred_wav_fg
        self.samples_pred_bg = self.net.pred_wav_bg
        
        print 'Restoring from:', pr.model_path
        if self.restore_only_shift:
          print 'restoring only shift'
          import tensorflow.contrib.slim as slim
          var_list = slim.get_variables_to_restore()
          var_list = [x for x in var_list if x.name.startswith('im/') or x.name.startswith('sf/') or x.name.startswith('joint/')]
          self.sess.run(tf.global_variables_initializer())
          tf.train.Saver(var_list).restore(self.sess, pr.model_path)
        else:
          tf.train.Saver().restore(self.sess, pr.model_path)
        tf.get_default_graph().finalize()

  def predict(self, ims, samples):
    print 'predict'
    print 'samples shape:', samples.shape
    spec_mix = self.sess.run(self.specgram_op, {self.samples_ph : samples})
    spec_pred_fg, spec_pred_bg, samples_pred_fg, samples_pred_bg = self.sess.run(
      [self.spec_pred_fg, self.spec_pred_bg, self.samples_pred_fg, self.samples_pred_bg], 
      {self.ims_ph : ims, self.samples_ph : samples})
    print 'samples pred shape:', samples.shape
    return dict(samples_pred_fg = samples_pred_fg, 
                samples_pred_bg = samples_pred_bg, 
                spec_pred_fg = spec_pred_fg, 
                spec_pred_bg = spec_pred_bg, 
                samples_mix = samples,
                spec_mix = spec_mix)

  def predict_unmixed(self, ims, samples0, samples1):
    # undo mixing
    samples_mix = samples0 + samples1
    spec_pred_fg, samples_pred_fg, spec_pred_bg, samples_pred_bg = self.sess.run(
      [self.spec_pred_fg, self.samples_pred_fg, self.spec_pred_bg, self.samples_pred_bg], 
      {self.ims_ph : ims[None], self.samples_ph : samples_mix[None]})
    spec0 = self.sess.run(self.specgram_op, {self.samples_ph : samples0[None]})
    spec1 = self.sess.run(self.specgram_op, {self.samples_ph : samples1[None]})
    spec_mix = self.sess.run(self.specgram_op, {self.samples_ph : samples_mix[None]})
    auto0 = self.sess.run(self.auto_op, {self.samples_ph : samples0[None]})
    auto1 = self.sess.run(self.auto_op, {self.samples_ph : samples1[None]})
    auto_mix = self.sess.run(self.auto_op, {self.samples_ph : samples_mix[None]})
    return dict(samples_pred_fg = samples_pred_fg[0],
                samples_pred_bg = samples_pred_bg[0],
                spec_pred_fg = spec_pred_fg[0],
                spec_pred_bg = spec_pred_bg[0],
                spec0 = spec0[0],
                spec1 = spec1[0], 
                spec_mix = spec_mix[0],
                auto_mix = auto_mix[0],
                auto0 = auto0[0],
                auto1 = auto1[0])
  
  #def predict_cam(self, ims, samples, n = 3, num_times = 3):
  def predict_cam(self, ims, samples, n = 5, num_times = 3):
    #num_times = 1
    if 1:
      f = min(ims.shape[1:3])
      ims = np.array([ig.scale(im, (f, f)) for im in ims])
      d = int(224./256 * ims.shape[1])
      print 'd =', d, ims.shape
      full = None
      count = None
      if n == 1:
        ys = [ims.shape[1]/2]
        xs = [ims.shape[2]/2]
      else:
        ys = np.linspace(0, ims.shape[1] - d, n).astype('int64')
        xs = np.linspace(0, ims.shape[2] - d, n).astype('int64')

      if num_times == 1:
        print 'Using one time'
        ts = [0.]
      else:
        ts = np.linspace(-2, 2., n)

      for y in ys:
        for x in xs:
          crop = ims[:, y : y + d, x : x + d]
          crop = resize_nd(crop, (crop.shape[0], pr.crop_im_dim, pr.crop_im_dim, 3), order = 1)
          for shift in ts:
            print x, y, t
            snd = sound.Sound(samples, self.pr.samp_sr)
            s0 = int(shift * snd.rate)
            s1 = s0 + snd.samples.shape[0]
            shifted = snd.pad_slice(s0, s1)
            assert shifted.samples.shape[0] == snd.samples.shape[0]

            [cam] = self.sess.run([self.net.vid_net.cam], 
                                  {self.ims_ph : crop[None], 
                                   self.samples_ph : shifted.samples[None]})
            cam = cam[0, ..., 0]
            if full is None:
              full = np.zeros(cam.shape[:1] + ims.shape[1:3])
              count = np.zeros_like(full)
            cam_resized = scipy.ndimage.zoom(
              cam, np.array((full.shape[0], d, d), 'float32') / np.array(cam.shape, 'float32'))
            if 1:
              print 'abs'
              cam_resized = np.abs(cam_resized)
            # print np.abs(cam_resized).max()

            frame0 = int(max(-shift, 0) * self.pr.fps)
            frame1 = cam_resized.shape[0] - int(max(shift, 0) * self.pr.fps)
            ok = np.ones(count.shape[0])
            cam_resized[:frame0] = 0.
            cam_resized[frame1:] = 0.
            ok[:frame0] = 0
            ok[frame1:] = 0

            full[:, y : y + d, x : x + d] += cam_resized
            count[:, y : y + d, x : x + d] += ok[:, None, None]
      assert count.min() > 0
      full /= np.maximum(count, 1e-5)
    #   ut.save('../results/full.pk', full)
    # full = ut.load('../results/full.pk')
    return full

def resize_nd(im, scale, order = 3):
  if np.ndim(scale) == 0:
    new_scale = [scale]*len(im.shape)
  elif type(scale[0]) == type(0):
    dims = scale
    new_scale = (np.array(dims, 'd') + 0.4) / np.array(im.shape, 'd')
    # a test to make sure we set the floating point scale correctly
    result_dims = map(int, new_scale * np.array(im.shape, 'd'))
    assert tuple(result_dims) == tuple(dims)
    scale_param = new_scale
  elif type(scale[0]) == type(0.) and type(scale[1]) == type(0.):
    new_scale = scale
  else:
    raise RuntimeError("don't know how to interpret scale: %s" % (scale,))
  res = scipy.ndimage.zoom(im, scale_param, order = order)
  # verify that zoom() returned an image of the desired size
  if (np.ndim(scale) != 0) and type(scale[0]) == type(0):
    assert res.shape == scale
  return res

def heatmap(frames, cam, lo_frac = 0.5, adapt = True, max_val = 35, videoin=None, frame_start=0):
  """ Set heatmap threshold adaptively, to deal with large variation in possible input videos. """
  frames = np.asarray(frames)
  max_prob = 0.35
  if adapt:
    max_val = np.percentile(cam, 97)

  same = np.max(cam) - np.min(cam) <= 0.001
  if same:
    return frames

  outs = []
  for i in xrange(frames.shape[0]):
    lo = lo_frac * max_val
    hi = max_val + 0.001
    im = frames[i]
    f = cam.shape[0] * float(i) / frames.shape[0]
    l = int(f)
    r = min(1 + l, cam.shape[0]-1)
    p = f - l
    frame_cam = ((1-p) * cam[l]) + (p * cam[r])
    frame_cam = ig.scale(frame_cam, im.shape[:2], 1)

    print "Frame cam after scale:", frame_cam.shape

    
    #vis = ut.cmap_im(pylab.cm.hot, np.minimum(frame_cam, hi), lo = lo, hi = hi)
    vis = ut.cmap_im(pylab.cm.jet, frame_cam, lo = lo, hi = hi)
    print "Fheatmap sahpe:", vis.shape

    # haha = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
    
    audiomask_dir = videoin.replace("segment", "audiomask")
    # audiomask_dir = videoin+"_audiomask"
    if not os.path.isdir(audiomask_dir):
      os.makedirs(audiomask_dir)

    if i % 4==0:
      cv2.imwrite(audiomask_dir + "/" + str(i+frame_start) + ".png", vis)
    # cv2.imshow("Heatmap1:", haha)

    #p = np.clip((frame_cam - lo)/float(hi - lo), 0, 1.)
    p = np.clip((frame_cam - lo)/float(hi - lo), 0, max_prob)
    p = p[..., None]
    im = np.array(im, 'd')
    # cv2.imshow("Heatmap:", vis)

    vis = np.array(vis, 'd')
    # cv2.waitKey(0)
    outs.append(np.uint8(im*(1-p) + vis*p))
  return np.array(outs)

def crop_from_cam(ims, cam, pr):
  cam = np.array([ig.blur(x, 2.) for x in cam])
  cam = np.abs(cam)
  cam = cam.mean(0)

  ims = np.asarray(ims)
  y, x = np.nonzero(cam >= cam.max() - 1e-8)
  y, x = y[0], x[0]
  y = int(round((y + 0.5) * ims.shape[1]/float(cam.shape[0])))
  x = int(round((x + 0.5) * ims.shape[2]/float(cam.shape[1])))

  d = np.mean(ims.shape[1:3])
  # h = int(max(224, d//3))
  # w = int(max(224, d//3))
  h = int(max(224, d//2.5))
  w = int(max(224, d//2.5))

  y0 = int(np.clip(y - h/2, 0, ims.shape[1] - h))
  x0 = int(np.clip(x - w/2, 0, ims.shape[2] - w))
  crop = ims[:, y0 : y0 + h, x0 : x0 + w]
  crop = np.array([ig.scale(im, (pr.crop_im_dim, pr.crop_im_dim)) for im in crop])
  return crop

def find_cam(ims, samples, arg, frame_start, fps):
  clf = shift_net.NetClf(
    shift_params.cam_v1(shift_dur = (0.5+len(ims))/float(fps), fps=fps), 
    '../results/nets/cam/net.tf-675000', gpu = arg.gpu)
  
  print "NUm images for cam prediction: ", len(ims)
  [cam] = clf.predict_cam_resize(ims[None], samples[None])
  cam = np.abs(cam[0, :, :, :, 0])
  print "Cam shape:---------------------------", cam.shape
  vis = heatmap(ims, cam, adapt = arg.adapt_cam_thresh, 
                max_val = arg.max_cam_thresh, videoin=arg.vid_file, frame_start=frame_start)
  return cam, vis

def run(vid_file, start_time, dur, pr, gpu, buf = 0.05, mask = None, arg = None, net = None):
  print pr
  buf=0
  dur = dur + buf

  # Devu added
  frame_start = int(start_time*pr.fps - arg.start*pr.fps)
  print "Here is frame strarttttttttt and end frame", frame_start, " :", frame_start+dur*pr.fps, 
  with ut.TmpDir() as vid_path:
    height_s = '-vf "scale=-2:\'min(%d,ih)\'"' % arg.max_full_height if arg.max_full_height > 0 else ''
    ut.sys_check(ut.frm(
      'ffmpeg -loglevel error -ss %(start_time)s -i "%(vid_file)s" -safe 0  '
      '-t %(dur)s -r %(pr.fps)s -vf scale=256:256 "%(vid_path)s/small_%%04d.png"'))
    ut.sys_check(ut.frm(
      'ffmpeg -loglevel error -ss %(start_time)s -i "%(vid_file)s" -safe 0 '
      '-t %(dur)s -r %(pr.fps)s %(height_s)s "%(vid_path)s/full_%%04d.png"'))
    ut.sys_check(ut.frm(
      'ffmpeg -loglevel error -ss %(start_time)s -i "%(vid_file)s" -safe 0  '
      '-t %(dur)s -ar %(pr.samp_sr)s -ac 2 "%(vid_path)s/sound.wav"'))

    if arg.fullres:
      fulls = map(ig.load, sorted(ut.glob(vid_path, 'full_*.png'))[:pr.sampled_frames])
      fulls = np.array(fulls)

    snd = sound.load_sound(pj(vid_path, 'sound.wav'))
    samples_orig = snd.normalized().samples
    samples_orig = samples_orig[:pr.num_samples]
    samples_src = samples_orig.copy()
    print "Samples src: ", samples_src.shape
    # sys.exit()

    print "sampled src: . num samples:", samples_src.shape, ", ", pr.num_samples
    print "sample per frame ", pr.samples_per_frame, "with fps : ", pr.fps
    print "sample frames in config: ", pr.sampled_frames
   
    if samples_src.shape[0] < pr.num_samples:
      return None
      
    ims = map(ig.load, sorted(ut.glob(vid_path, 'small_*.png')))
    ims = np.array(ims)
    d = 224
    y = x = ims.shape[1]/2 - d/2
    ims = ims[:, y : y + d, x : x + d]
    ims = ims[:pr.sampled_frames]
    # DEVU add
    # pr.sampled_frames = int(np.ceil(dur*pr.fps))
  

    if mask == 'l':
      ims[:, :, :ims.shape[2]/2] = 128
      if arg.fullres:
        fulls[:, :, :fulls.shape[2]/2] = 128
    elif mask == 'r':
      ims[:, :, ims.shape[2]/2:] = 128
      if arg.fullres:
        fulls[:, :, fulls.shape[2]/2:] = 128
    elif mask is None:
      pass
    else: raise RuntimeError()

    samples_src = mu.normalize_rms_np(samples_src[None], pr.input_rms)[0]
    net.init()
    ret = net.predict(ims[None], samples_src[None])
    samples_pred_fg = ret['samples_pred_fg'][0][:, None]
    samples_pred_bg = ret['samples_pred_bg'][0][:, None]
    spec_pred_fg = ret['spec_pred_fg'][0]
    spec_pred_bg = ret['spec_pred_bg'][0]
    print spec_pred_bg.shape
    spec_mix = ret['spec_mix'][0]


    if arg.cam:
      cam, vis = find_cam(fulls, samples_orig, arg, frame_start, fps=pr.fps) # new fps devu added
    else:
      if arg.fullres:
        vis = fulls
      else:
        vis = ims

    return dict(ims = vis, 
                samples_pred_fg = samples_pred_fg, 
                samples_pred_bg = samples_pred_bg, 
                samples_mix = ret['samples_mix'][0],
                samples_src = samples_src, 
                spec_pred_fg = spec_pred_fg, 
                spec_pred_bg = spec_pred_bg, 
                spec_mix = spec_mix)
    
def get_duration_video_file(f,fps):
  cap = cv2.VideoCapture(f)
  num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  if fps is None:
    fps = cap.get(cv2.CAP_PROP_FPS)
  

  return num_frames/float(fps)

def get_fps_video_file(f):
  cap = cv2.VideoCapture(f)
  fps = cap.get(cv2.CAP_PROP_FPS)

  return fps


if __name__ == '__main__':
  arg = argparse.ArgumentParser(description='Separate on- and off-screen audio from a video')
  arg.add_argument('vid_file', type = str, help = 'Video file to process')
  arg.add_argument('--duration_mult', type = float, default = None, 
                   help = 'Multiply the default duration of the audio (i.e. %f) by this amount. Should be a power of 2.' % sep_params.VidDur)
  arg.add_argument('--mask', type = str, default = None, 
                   help = "set to 'l' or 'r' to visually mask the left/right half of the video before processing")
  arg.add_argument('--start', type = float, default = 0., help = 'How many seconds into the video to start')
  arg.add_argument('--model', type = str, default = 'full', 
                   help = 'Which variation of othe source separation model to run.')
  arg.add_argument('--gpu', type = int, default = 0, help = 'Set to -1 for no GPU')
  arg.add_argument('--out', type = str, default = None, help = 'Directory to save videos')
  arg.add_argument('--cam', dest = 'cam', default = False, action = 'store_true')
  arg.add_argument('--adapt_cam_thresh', type = int, default = True)
  arg.add_argument('--max_cam_thresh', type = float, default = 35)

  # undocumented/deprecated options
  arg.add_argument('--clip_dur', type = float, default = None)
  arg.add_argument('--duration', type = float, default = None)
  arg.add_argument('--fullres', type = bool, default = True)
  arg.add_argument('--suffix', type = str, default = '')
  arg.add_argument('--max_full_height', type = int, default = 600)

  # Our customized params
  arg.add_argument('--videosegment_dir', type = str, default = "/media/Databases/preprocess_avspeech/segment", help = 'Directory to video segemnt')
  arg.add_argument('--start_clip_index', type = int, default = 6, help = 'Sart clip index')
  arg.add_argument('--n_process', type = int, default = 16, help = 'NUmber of process for parallell processing')

  #arg.set_defaults(cam = False)

  ##### Common set up for all processes
  arg = arg.parse_args()
  arg.fullres = arg.fullres or arg.cam

  if arg.gpu < 0:
    arg.gpu = None

  print 'Start time:', arg.start
  print 'GPU =', arg.gpu

  gpus = [arg.gpu]
  gpus = mu.set_gpus(gpus)

  import glob
  import pandas as pd

  data = arg.videosegment_dir # "/media/Databases/preprocess_avspeech/segment"


  clips = list(pd.read_csv("2faces_current.txt", header=None)[0])
  files = []
  """
  for video in videos:
    clips = glob.glob(data+"/"+video+"/"+"*.mp4")
    clips = clips[:2]
    files += clips
  """
  
  # Processed files . not removing empty folder
  output = data.replace("segment", "audiomask")
  if os.path.isdir(output):
    processed_files = glob.glob(output + "/*/*.mp4")
    processed_files = [f.replace("audiomask", "segment") for f in processed_files]
  else:
    processed_files = []

  def count_empty_folder(dir):
      empty_denoised_folder = []
      folders = glob.glob(dir+"/*/*.mp4")
      i = 0
      for fo in folders:
          print(fo)
          jpgs = glob.glob(fo + "/*.png")
          print(len(jpgs))
          if len(jpgs)==0:
              empty_denoised_folder.append(fo)
      return empty_denoised_folder


  empty_outputs = count_empty_folder(output)
  empty_files = [i.replace("audiomask", "segment")for i in empty_outputs]
  


  #print(output)
  # for clip in clips:
  #   clips = [f for f in os.listdir(data +"/" + video) if f.endswith('.mp4')]
  #   clips = [data + "/" + video +"/"+f for f in clips]
  #   clips = clips[int(arg.start_clip_index):]
  #   files += clips

  files = [data+"/"+c  for c in clips]

  #print processed_files[0], files[0]
  print "Processed files len: ", len(processed_files)
  print "Total files: ", len(files)
  print "Empty files:", len(empty_files)
  files = list(set(files)-set(processed_files))
  files =set(files).union(empty_files)

  print "Will process only : ", len(files)
  
  arg_original = copy.deepcopy(arg)

  def process_and_generate_audio_mask(f):
    try:
      arg = copy.deepcopy(arg_original)
      duration = get_duration_video_file(f,None)
      arg.duration = duration
      arg.vid_file = f
      if arg.duration_mult is not None:
        pr = sep_params.full()
        step = 0.001 * pr.frame_step_ms
        length = 0.001 * pr.frame_length_ms
        arg.clip_dur = length + step*(0.5+pr.spec_len)*arg.duration_mult

      fn = getattr(sep_params, arg.model)
      pr = fn(vid_dur = arg.clip_dur, fps=get_fps_video_file(f))
      print "Real fps and pr fps: ", get_fps_video_file(f), ".pr:", pr.fps
      print "real duration.  arg.clip_dur: ", duration, ". ",  arg.clip_dur
      print "Duration mul: ", arg.duration_mult


      if arg.clip_dur is None:
        arg.clip_dur = pr.vid_dur

      pr.input_rms = np.sqrt(0.1**2 + 0.1**2)
      pr.model_path = '../results/nets/sep/%s/net.tf-%d' % (pr.name, pr.train_iters)

      if not os.path.exists(arg.vid_file):
        print 'Does not exist:', arg.vid_file
        #sys.exit(1)
        return

      if arg.duration is None:
        arg.duration = arg.clip_dur + 0.01

      print arg.duration, arg.clip_dur
      full_dur = arg.duration
      #full_dur = min(arg.duration, ut.video_length(arg.vid_file))
      #full_dur = arg.duration
      step_dur = arg.clip_dur/2.
      filled = np.zeros(int(np.ceil(full_dur * pr.samp_sr)), 'bool')
      full_samples_fg = np.zeros(filled.shape, 'float32')
      full_samples_bg = np.zeros(filled.shape, 'float32')
      full_samples_src = np.zeros(filled.shape, 'float32')
      arg.start = ut.make_mod(arg.start, (1./pr.fps))

      print "Full dur, clip dur: ", full_dur, ". ", arg.clip_dur
      ts = np.arange(arg.start, arg.start + full_dur-arg.clip_dur, arg.clip_dur)
      if len(ts)==1:
        ts = np.append(ts, [full_dur-arg.clip_dur])
      # ts= [0, arg.clip_dur]
      print "Ts:", ts
      # sys.exit()
      full_ims = [None] * int(np.ceil(full_dur * pr.fps))
      print "Full imgs:", len(full_ims)
      # sys.exit()

      net = NetClf(pr, gpu = gpus[0])

      for t in ut.time_est(ts):
        t = ut.make_mod(t, (1./pr.fps))
        frame_start = int(t*pr.fps - arg.start*pr.fps)
        print "Duration of segment : ", arg.clip_dur
        ret = run(arg.vid_file, t, arg.clip_dur, pr, gpus[0], mask = arg.mask, arg = arg, net = net)
        
        if ret is None:
          print("Noneeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
          continue
        ims = ret['ims']
        print "Here is len of ims:", len(ims)
    except Exception as e:
      log =  "There is error when processing file : " + str(arg.vid_file) + "\n" + "Here is detail excpetion: " + str(e) + "\n"
      write_log(log)
      print log    
      traceback.print_exc()   




#print("Len files: ", len(files))
#print("File 0: ", files[0])

# BAckup old log file


def pool_handler(files):
    n_process = int(arg_original.n_process)
    p = Pool(n_process)
    p.map(process_and_generate_audio_mask, files)

#files = [files[0], files[0], files[0], files[0], files[0], files[0], files[0], files[0]]
# if os.path.isfile("error.txt"):
#   os.system('mv error.txt' + ' error_' + get_current_time() +'.txt')
# pool_handler(files)


"""
  for f in files[:2]:
    print "This is file----------------------------------------", f

    # f = "/home/vuthede/data/segment_clean/Oxymoron Antithesis Paradox/227.200000_230.960000.mp4"
    # f = "/home/vuthede/data/segment_clean/Oxymoron Antithesis Paradox/215.520000_226.320000.mp4"

    f = "/home/vuthede/data/segment_clean/Catherine Steiner-Adair How Technology Affects Child Development/90.131711_93.176422.mp4"
    duration = get_duration_video_file(f,None)
    print "Duration haha:", duration
    # arg.duration_mult = np.ceil(duration/sep_params.VidDur) 
    arg.duration = duration
    arg.vid_file = f
 
    if arg.duration_mult is not None:
      pr = sep_params.full()
      step = 0.001 * pr.frame_step_ms
      length = 0.001 * pr.frame_length_ms
      arg.clip_dur = length + step*(0.5+pr.spec_len)*arg.duration_mult
    
    fn = getattr(sep_params, arg.model)
    pr = fn(vid_dur = arg.clip_dur, fps=get_fps_video_file(f))
    print "Real fps and pr fps: ", get_fps_video_file(f), ".pr:", pr.fps
    print "real duration.  arg.clip_dur: ", duration, ". ",  arg.clip_dur
    print "Duration mul: ", arg.duration_mult
    # sys.exit()
    #Customized some params by Devu
    


    if arg.clip_dur is None:
      arg.clip_dur = pr.vid_dur
    pr.input_rms = np.sqrt(0.1**2 + 0.1**2)
    print 'Spectrogram samples:', pr.spec_len
    pr.model_path = '../results/nets/sep/%s/net.tf-%d' % (pr.name, pr.train_iters)

    if not os.path.exists(arg.vid_file):
      print 'Does not exist:', arg.vid_file
      sys.exit(1)

    if arg.duration is None:
      arg.duration = arg.clip_dur + 0.01

    print arg.duration, arg.clip_dur
    full_dur = arg.duration
    #full_dur = min(arg.duration, ut.video_length(arg.vid_file))
    #full_dur = arg.duration
    step_dur = arg.clip_dur/2.
    filled = np.zeros(int(np.ceil(full_dur * pr.samp_sr)), 'bool')
    full_samples_fg = np.zeros(filled.shape, 'float32')
    full_samples_bg = np.zeros(filled.shape, 'float32')
    full_samples_src = np.zeros(filled.shape, 'float32')
    arg.start = ut.make_mod(arg.start, (1./pr.fps))

    print "Full dur, clip dur: ", full_dur, ". ", arg.clip_dur
    ts = np.arange(arg.start, arg.start + full_dur-arg.clip_dur, arg.clip_dur)
    if len(ts)==1:
      ts = np.append(ts, [full_dur-arg.clip_dur])
    # ts= [0, arg.clip_dur]
    print "Ts:", ts
    # sys.exit()
    full_ims = [None] * int(np.ceil(full_dur * pr.fps))
    print "Full imgs:", len(full_ims)
    # sys.exit()

    net = NetClf(pr, gpu = gpus[0])

    for t in ut.time_est(ts):
      t = ut.make_mod(t, (1./pr.fps))
      frame_start = int(t*pr.fps - arg.start*pr.fps)
      print "Duration of segment : ", arg.clip_dur
      ret = run(arg.vid_file, t, arg.clip_dur, pr, gpus[0], mask = arg.mask, arg = arg, net = net)
      if ret is None:
        print("Noneeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
        continue
      ims = ret['ims']
      print "Here is len of ims:", len(ims)
"""
