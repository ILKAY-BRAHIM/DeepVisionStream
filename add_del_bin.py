import sys
sys.path.append('/opt/nvidia/deepstream/deepstream-7.1/sources/deepstream_python_apps/apps/common')

import gi
import configparser
import time
import math
import numpy as np
import pyds
import os
import json
from platform_info import PlatformInfo
from bus_call import bus_call
from FPS import PERF_DATA
import FPS 
from gi.repository import GLib, Gst

Gst.init(None)

gi.require_version('Gst', '1.0')

class DeepStreamMultiStream:
    def __init__(self):
        self.PGIE_CLASS_ID_VEHICLE = 0
        self.MUXER_OUTPUT_WIDTH = 1920
        self.MUXER_OUTPUT_HEIGHT = 1080
        self.TILED_OUTPUT_WIDTH = 1920
        self.TILED_OUTPUT_HEIGHT = 1080
        self.GST_CAPS_FEATURES_NVMM = "memory:NVMM"

        self.perf_data = PERF_DATA()
        self.frame_count = {}
        self.saved_count = {}
        self.EXTRACTED_DATA = []
        self.platform_info = PlatformInfo()
        self.folder_name = "output"
        self.pipeline = None
        self.streammux = None
        self.pgie = None
        self.tiler = None
        self.nvvidconv = None
        self.nvosd = None
        self.sink = None
        self.source_bins = {}
        self.num_sources = 0
        self.g_eos_list = {}
        self.loop = None

    def cleanup(self):
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        if self.loop:
            self.loop.quit()

    def cb_newpad_for_bin(self, decodebin, pad, bin):
        caps = pad.get_current_caps()
        if not caps:
            return
        name = caps.get_structure(0).get_name()
        features = caps.get_features(0)

        if "video" in name:
            if features.contains(self.GST_CAPS_FEATURES_NVMM):
                ghost_pad = bin.get_static_pad("src")
                ghost_pad.set_target(pad)
            else:
                print("Non-NVMM memory, decoder not supported")

    def decodebin_child_added(self, child_proxy, obj, name, user_data):
        if "decodebin" in name:
            obj.connect("child-added", self.decodebin_child_added, user_data)
        if "nvv4l2decoder" in name:
            if self.platform_info.is_integrated_gpu():
                obj.set_property("enable-max-performance", True)
                obj.set_property("drop-frame-interval", 0)
                obj.set_property("num-extra-surfaces", 0)
            else:
                obj.set_property("gpu_id", 0)

    def creat_bin(self, index, uri):
        bin_name = f"source-bin-{index}"
        bin = Gst.Bin.new(bin_name)

        uri_decodebin = Gst.ElementFactory.make("uridecodebin", f"uri-decode-bin-{index}")
        uri_decodebin.set_property("uri", uri)
        uri_decodebin.connect("pad-added", self.cb_newpad_for_bin, bin)
        uri_decodebin.connect("child-added", self.decodebin_child_added, bin)

        bin.add(uri_decodebin)
        ghost_pad = Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC)
        bin.add_pad(ghost_pad)

        return bin

    def add_sources(self, uri):
        source_id = self.num_sources
        source_bin = self.creat_bin(source_id, uri)
        self.pipeline.add(source_bin)
        src_pad = source_bin.get_static_pad("src")
        sink_pad = self.streammux.request_pad_simple(f"sink_{source_id}")
        src_pad.link(sink_pad)
        source_bin.set_state(Gst.State.PLAYING)
        self.source_bins[source_id] = source_bin
        self.num_sources += 1
        return source_id

    def stop_release_source(self, id_source):
        self.source_bins[id_source].set_state(Gst.State.NULL)
        pad_name = f"sink_{id_source}"
        sinkpad = self.streammux.get_static_pad(pad_name)
        if sinkpad:
            sinkpad.send_event(Gst.Event.new_flush_stop(False))
            self.streammux.release_request_pad(sinkpad)
        self.pipeline.remove(self.source_bins[id_source])
        self.num_sources -= 1

    def delete_sources(self, id_source):
        self.stop_release_source(id_source)
        if self.num_sources == 0:
            self.loop.quit()
        return True

    def bus_call(self, bus, message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("End-of-stream")
            self.loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print("Error:", err, debug)
            self.loop.quit()
        return True

    def perf_print_callback(self):
        self.perf_data.perf_print_callback()
        return True
    
    def tiler_sink_pad_buffer_probe(self, pad, info, u_data):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            print("Unable to get GstBuffer")
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list

        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            stream_id = f"stream{frame_meta.pad_index}"

            # ✅ Fix: Instantiate GETFPS with required stream_id
            if stream_id not in self.perf_data.all_stream_fps:
                from FPS import GETFPS
                self.perf_data.all_stream_fps[stream_id] = GETFPS(stream_id)

            self.perf_data.update_fps(stream_id)

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    def start(self, video_path):
        self.video_path = video_path
        self.pipeline = Gst.Pipeline()

        self.streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
        self.streammux.set_property("batched-push-timeout", 25000)
        self.streammux.set_property("batch-size", 120)
        self.streammux.set_property("gpu-id", 0)
        self.streammux.set_property("width", self.MUXER_OUTPUT_WIDTH)
        self.streammux.set_property("height", self.MUXER_OUTPUT_HEIGHT)
        self.pipeline.add(self.streammux)

        self.pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
        self.pgie.set_property("config-file-path", "./config_infer_primary_yolo11.txt")
        self.pgie.set_property("batch-size", 16)
        self.pgie.set_property("gpu-id", 0)

        self.tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
        self.tiler.set_property("rows", 1)
        self.tiler.set_property("columns", 2)
        self.tiler.set_property("width", self.TILED_OUTPUT_WIDTH)
        self.tiler.set_property("height", self.TILED_OUTPUT_HEIGHT)
        self.tiler.set_property("gpu_id", 0)

        self.nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        self.nvvidconv.set_property("gpu_id", 0)

        self.nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        self.nvosd.set_property("gpu_id", 0)

        self.sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        self.sink.set_property("sync", 0)
        self.sink.set_property("qos", 0)

        for elem in [self.pgie, self.tiler, self.nvvidconv, self.nvosd, self.sink]:
            self.pipeline.add(elem)

        self.streammux.link(self.pgie)
        self.pgie.link(self.tiler)
        self.tiler.link(self.nvvidconv)
        self.nvvidconv.link(self.nvosd)
        self.nvosd.link(self.sink)

        self.loop = GLib.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call, self.loop)

        # self.tiler.get_static_pad("src").add_probe(Gst.PadProbeType.EVENT_DOWNSTREAM, self.tiler_sink_pad_buffer_probe, 0)
        self.tiler.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, self.tiler_sink_pad_buffer_probe, 0)


        GLib.timeout_add(5000, self.perf_data.perf_print_callback)

        self.pipeline.set_state(Gst.State.PLAYING)

        try:
            self.loop.run()
        except:
            pass

        self.pipeline.set_state(Gst.State.NULL)


def main():
    video_path = "file:///opt/nvidia/deepstream/deepstream-7.1/sources/my_data/best.mp4"
    deepstream_list = []
    for i in range(10):
        deepstream_list.append(DeepStreamMultiStream())

    import threading
    for deepstream in deepstream_list:
        thread = threading.Thread(target=deepstream.start, args=(video_path,))
        thread.start()
        time.sleep(1)
    all_ids = []
    for deepstream in deepstream_list:
        ids = []
        for i in range(6):
            ids.append(deepstream.add_sources(video_path))
            time.sleep(1)
        all_ids.append(ids)

    for ids in all_ids:
        for id in ids:
            for deepstream in deepstream_list:
                deepstream.delete_sources(id)
                time.sleep(1)
if __name__ == "__main__":
    main()
