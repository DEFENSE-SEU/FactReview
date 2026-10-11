"""Disposable CPU inference collector; imports Torch only inside the actual runner.

This collector observes an unchanged author execution. It provides source-linked
facts, with no scientific grant and no adversarial in-process guarantee.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import runpy
import sys
import traceback
import types
from pathlib import Path

VERSION = "torch-cpu-inference-v1"


def digest_file(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            result.update(chunk)
    return result.hexdigest()


def select(value, keys):
    for key in keys:
        if not ((type(value) is dict and type(key) is str and key in value) or
                (type(value) in (list,tuple) and type(key) is int and 0 <= key < len(value))):
            raise ValueError("unsupported_runtime_selector")
        value = value[key]
    return value


def run(config):
    report = {"version":VERSION,"binding_digest":config["inference"]["binding_digest"],
              "unresolved":[],"calls":[],"samples":[],"execution":{"status":"not_started"}}
    restore, handles = [], []
    prior_profile, prior_argv, prior_path, prior_cwd = sys.getprofile(),sys.argv[:],sys.path[:],os.getcwd()
    failure = audit_failure = None
    author_started = False
    audit_enabled = [False]

    def restore_observation():
        audit_enabled[0]=False
        sys.setprofile(prior_profile)
        while restore:
            owner,name,owned,original=restore.pop()
            if owned:
                setattr(owner,name,original)
            else:
                delattr(owner,name)
        while handles:
            handles.pop().remove()
        sys.argv,sys.path=prior_argv[:],prior_path[:]
        os.chdir(prior_cwd)

    def author_environment():
        os.chdir(config["cwd"])
        sys.argv=[config["entry"],*config["args"]]
        sys.path.insert(0,str(Path(config["entry"]).resolve().parent))
    try:
        import torch
        from source_binding import _code_sha, prepare_site
        nn = config["inference"]
        proposal, policy = nn["proposal"], nn["policy"]
        nn["rows"]=[]
        partition="/app/"+proposal["dataset_path"]
        if digest_file(partition)!=nn["partition_sha256"]:
            raise ValueError("partition_identity_changed")
        byte_count=0
        with open(partition,"rb") as stream:
            while raw:=stream.readline(policy["max_jsonl_line_bytes"]+1):
                byte_count+=len(raw)
                if (len(raw)>policy["max_jsonl_line_bytes"] or len(nn["rows"])>=policy["max_samples"]
                        or byte_count>policy["max_partition_bytes"]):
                    raise ValueError("partition_runtime_capacity")
                item=json.loads(raw)
                nn["rows"].append({"ordinal":len(nn["rows"]),"id":item[proposal["id_key"]],
                    "features":item[proposal["features_key"]],"label":item[proposal["label_key"]]})
        if torch.__version__.split("+")[0] != policy["torch_version"] or prior_profile is not None:
            raise ValueError("framework_version_or_profile_unsupported")
        package = Path(torch.__file__).parent
        approved_python={str(Path("/app/"+key).resolve()) for key in nn["source_sha256"] if key.endswith(".py")}
        observer_sources={str(Path(__file__).resolve()),str(Path(__file__).with_name("source_binding.py").resolve())}
        stdlib_root=Path(os.__file__).resolve().parent
        forbidden_stdlib={"ctypes","_ctypes","importlib","subprocess","multiprocessing","asyncio","concurrent"}
        approved_names=(set(sys.stdlib_module_names)-forbidden_stdlib)|{"torch"}
        local_imports={part for key in nn["source_sha256"] if key.endswith(".py")
                       for part in (Path(key).parts[0],Path(key).stem) if part != "__init__"}
        def framework_sha():
            manifest=hashlib.sha256()
            for path in sorted(package.rglob("*")):
                if path.is_file() and path.suffix in {".py",".so",".pyd",".dll"}:
                    manifest.update(path.relative_to(package).as_posix().encode()+b"\0"+digest_file(path).encode()+b"\n")
            return manifest.hexdigest()
        initial_framework_sha=framework_sha()
        if initial_framework_sha != policy["package_manifest_sha256"]:
            raise ValueError("runtime_framework_package_identity_unapproved")
        report["framework"]={"version":torch.__version__,"package_manifest_sha256":initial_framework_sha,"device":"cpu"}
        report["interpreter"]={"version":sys.version,"optimize":sys.flags.optimize}
        report["entry"]={"argv":[config["entry"],*config["args"]],"cwd":config["cwd"],
                         "path":config["cwd"]+"/"+config["entry"],"sha256":digest_file(config["cwd"]+"/"+config["entry"])}
        report["source_hashes_before"]={"/app/"+key:digest_file("/app/"+key) for key in nn["source_sha256"]}
        sites=[prepare_site(s["path"],s["qualname"],s["firstlineno"]) for s in config["sites"]]
        if any(actual["sha256"] != supplied["sha256"] for actual,supplied in zip(sites,config["sites"],strict=True)):
            raise ValueError("runtime_source_identity_changed")
        report["sites"]=sites
        tokens, held, pending, completed, stack = {},[],[],[],[]
        state, loaded, batch, active = {},{},None,{}
        checkpoint = None
        checkpoint_state = None
        observing = False
        counter = 0
        captured_elements = 0
        snapshot_cache = {}

        def unknown(reason):
            if reason not in report["unresolved"]:
                report["unresolved"].append(reason)

        def capture(callback):
            nonlocal observing
            previous, observing = observing, True
            try:
                callback()
            except Exception as exc:
                unknown(str(exc) if type(exc) is ValueError else "capture_failed_"+type(exc).__name__)
            finally:
                observing = previous

        def token(value):
            key=id(value)
            if key not in tokens:
                if len(tokens) >= policy["max_events"] * 8:
                    raise ValueError("object_identity_capacity")
                tokens[key]=len(tokens)+1
                held.append(value)
            return tokens[key]

        def snapshot(value, *, values=True):
            nonlocal captured_elements
            if type(value) not in (torch.Tensor,torch.nn.Parameter) or value.device.type != "cpu" or value.layout != torch.strided or not value.is_contiguous():
                raise ValueError("unsupported_tensor_layout_or_type")
            if value.numel() > policy["max_tensor_elements"] and values:
                raise ValueError("actual_tensor_snapshot_capacity")
            if len(value.shape)>8:
                raise ValueError("actual_tensor_rank_capacity")
            raw=value.detach().reshape(-1).view(torch.uint8)
            sha=hashlib.sha256()
            for offset in range(0,raw.numel(),1048576):
                sha.update(bytes(raw[offset:offset+1048576].tolist()))
            key=(token(value),value._version,values,sha.hexdigest())
            if values and key in snapshot_cache:
                return snapshot_cache[key]
            if values:
                captured_elements += value.numel()
                if captured_elements > policy["max_tensor_elements"]:
                    raise ValueError("whole_run_tensor_snapshot_capacity")
            result={"token":token(value),"shape":list(value.shape),"dtype":str(value.dtype),"sha256":sha.hexdigest(),"tensor_version":value._version}
            if values:
                result["values"]=value.detach().tolist()
                def finite(item):
                    if type(item) is list:
                        return all(finite(child) for child in item)
                    return type(item) in (int,float) and math.isfinite(item)
                if not finite(result["values"]):
                    raise ValueError("unsupported_or_nonfinite_tensor_values")
            if values:
                snapshot_cache[key]=result
            return result

        def module_state(module):
            rows={}
            visiting=set()
            def visit(current,prefix):
                if id(current) in visiting or len(rows) > policy["max_events"]:
                    raise ValueError("registered_state_cycle_or_capacity")
                visiting.add(id(current))
                namespace=object.__getattribute__(current,"__dict__")
                if any(namespace.get(key) for key in ("_forward_pre_hooks","_forward_hooks","_load_state_dict_pre_hooks","_load_state_dict_post_hooks")):
                    raise ValueError("preexisting_registered_module_hooks_uncovered")
                if type(namespace.get("_parameters")) is not dict or type(namespace.get("_buffers")) is not dict:
                    raise ValueError("nonstandard_registered_state")
                for key,value in namespace["_parameters"].items():
                    if value is not None:
                        rows[prefix+key]={**snapshot(value,values=False),"checkpoint_member":True}
                nonpersistent=namespace.get("_non_persistent_buffers_set",set())
                for key,value in namespace["_buffers"].items():
                    if value is not None:
                        rows[prefix+key]={**snapshot(value,values=False),"checkpoint_member":key not in nonpersistent}
                for key,child in namespace["_modules"].items():
                    if child is not None:
                        visit(child,prefix+key+".")
                visiting.remove(id(current))
            visit(module,"")
            return {name:{k:v for k,v in row.items() if k != "token"} for name,row in rows.items()}

        def patch(owner,name,replacement):
            restore.append((owner,name,name in vars(owner),getattr(owner,name)))
            setattr(owner,name,replacement)

        original_load=torch.load
        def load(*args,**kwargs):
            nonlocal checkpoint
            value=original_load(*args,**kwargs)
            def record():
                nonlocal checkpoint,checkpoint_state
                if not args or type(args[0]) is not str or str(Path(args[0]).resolve()) != "/app/"+proposal["checkpoint_path"]:
                    return
                if kwargs.get("weights_only",True) is not True:
                    raise ValueError("checkpoint_weights_only_required")
                if digest_file(args[0]) != nn["checkpoint_sha256"]:
                    raise ValueError("checkpoint_changed")
                checkpoint=select(value,proposal["state_selector"])
                if type(checkpoint) is not dict:
                    from collections import OrderedDict
                    if type(checkpoint) is not OrderedDict:
                        raise ValueError("unsupported_checkpoint_map")
                if any(type(key) is not str or not key for key in checkpoint):
                    raise ValueError("checkpoint_tensor_keys_required")
                checkpoint_state={key:{field:item for field,item in snapshot(value,values=False).items()
                    if field not in {"token","tensor_version"}} for key,value in checkpoint.items()}
                report["checkpoint_sha256"]=nn["checkpoint_sha256"]
            capture(record)
            return value
        patch(torch,"load",load)
        original_state=torch.nn.Module.load_state_dict
        def load_state(module,mapping,*args,**kwargs):
            value=original_state(module,mapping,*args,**kwargs)
            def record():
                if mapping is not checkpoint or checkpoint is None:
                    return
                expected={k:{x:y for x,y in snapshot(v,values=False).items() if x != "token"} for k,v in mapping.items()}
                # Version counters are local to storage; compare original bytes/dtype/shape.
                expected={k:{x:y for x,y in v.items() if x != "tensor_version"} for k,v in expected.items()}
                actual=module_state(module)
                comparable={k:{x:y for x,y in v.items() if x not in {"tensor_version","checkpoint_member"}}
                            for k,v in actual.items() if v["checkpoint_member"]}
                if comparable != expected or expected != checkpoint_state or value.missing_keys or value.unexpected_keys:
                    raise ValueError("checkpoint_loaded_state_mismatch")
                state.update(actual)
                loaded["module"]=module
                report["loaded_model_token"]=token(module)
                report["loaded_state"]=actual
            capture(record)
            return value
        patch(torch.nn.Module,"load_state_dict",load_state)

        original_call=torch.nn.Module._call_impl
        def module_call(module,*args,**kwargs):
            chosen=loaded.get("module") is module
            if chosen:
                loaded["call_active"]=True
                loaded["final_forward"]=None
                def hooks():
                    namespace=object.__getattribute__(module,"__dict__")
                    if any(namespace.get(key) for key in ("_forward_pre_hooks","_forward_hooks")):
                        raise ValueError("preexisting_module_hooks_uncovered")
                    module_globals=torch.nn.modules.module
                    if module_globals._global_forward_hooks or module_globals._global_forward_pre_hooks:
                        raise ValueError("global_module_hooks_uncovered")
                capture(hooks)
            try:
                value=original_call(module,*args,**kwargs)
                if chosen:
                    def record():
                        row=loaded.get("final_forward")
                        if row is None or snapshot(value)!=row["output"]:
                            raise ValueError("final_module_output_identity_unobserved")
                        row["module_call_completed"]=True
                    capture(record)
                return value
            finally:
                if chosen:
                    loaded["call_active"]=False
        patch(torch.nn.Module,"_call_impl",module_call)

        original_next=torch.utils.data.dataloader._SingleProcessDataLoaderIter.__next__
        def next_batch(iterator):
            nonlocal batch
            pending.clear()
            value=original_next(iterator)
            batch=None
            def record():
                nonlocal batch
                if not pending:
                    raise ValueError("unobserved_getitem_batch")
                if iterator._num_workers != 0 or iterator._drop_last or iterator._collate_fn is not torch.utils.data.default_collate:
                    raise ValueError("unsupported_loader_policy")
                features=select(value,proposal["batch_features_selector"])
                labels=select(value,proposal["batch_labels_selector"])
                fs,ls=snapshot(features),snapshot(labels)
                identifiers=select(value,proposal["batch_ids_selector"])
                if type(identifiers) is torch.Tensor:
                    identifiers=snapshot(identifiers)["values"]
                if type(identifiers) not in (list,tuple) or any(type(value) not in (str,int) for value in identifiers):
                    raise ValueError("actual_batch_ids_unavailable")
                if len(fs["values"]) != len(pending) or len(ls["values"]) != len(pending):
                    raise ValueError("batch_size_getitem_mismatch")
                if len(identifiers)!=len(pending) or any(type(a) is not type(row["id"]) or a!=row["id"]
                        for a,row in zip(identifiers,pending,strict=True)):
                    raise ValueError("sample_batch_id_binding_changed")
                batch={"features":features,"labels":labels,"ids":list(identifiers),"rows":list(pending),"feature_snapshot":fs,"label_snapshot":ls}
            capture(record)
            return value
        patch(torch.utils.data.dataloader._SingleProcessDataLoaderIter,"__next__",next_batch)
        # Unobserved Python concurrency cannot supply a complete ordered receipt.
        import _thread
        import threading
        original_thread=_thread.start_new_thread
        def start_thread(*args,**kwargs):
            unknown("python_thread_execution_uncovered")
            return original_thread(*args,**kwargs)
        patch(_thread,"start_new_thread",start_thread)
        original_start=threading.Thread.start
        def thread_start(*args,**kwargs):
            unknown("python_thread_execution_uncovered")
            return original_start(*args,**kwargs)
        patch(threading.Thread,"start",thread_start)
        for name in ("compile",):
            original=getattr(torch,name)
            def unsupported(*args,_original=original,**kwargs):
                unknown("compiled_model_execution_uncovered")
                return _original(*args,**kwargs)
            patch(torch,name,unsupported)
        # Preserve the original operation; unknown extension registration cannot qualify.
        for owner,name in ((torch.ops,"load_library"),(torch.library.Library,"__init__"),
                           (torch.library,"custom_op"),(torch,"from_numpy"),(torch._C,"_dispatch_library")):
            original=getattr(owner,name)
            def foreign_loader(*args,_original=original,**kwargs):
                unknown("unapproved_extension_or_foreign_array_operation")
                return _original(*args,**kwargs)
            patch(owner,name,foreign_loader)
        def audit(event,args):
            if not audit_enabled[0]:
                return
            if event in {"subprocess.Popen","os.fork","os.posix_spawn","os.exec","os.system"}:
                unknown("child_process_execution_uncovered")
            if event in {"ctypes.dlopen","ctypes.dlsym"}:
                unknown("unapproved_dynamic_native_load")
            if event=="import" and (not args or type(args[0]) is not str or args[0].split(".",1)[0] not in approved_names|local_imports):
                unknown("unapproved_runtime_import")
        sys.addaudithook(audit)  # Disposable CLI only; never exposed as a host API.
        audit_enabled[0]=True

        def python_origin(filename):
            if filename in approved_python or filename in observer_sources:
                return True
            if filename.startswith("<frozen "):
                # Bootstrap imports are machinery; imported code has its own origin check.
                root=filename.removeprefix("<frozen ").split(".",1)[0].rstrip(">")
                return root in approved_names|{"importlib"}
            path=Path(filename)
            if not path.is_absolute():
                return False
            if path.is_relative_to(package):
                relative=path.relative_to(package)
                return path.suffix==".py" and (relative.as_posix()=="jit/_builtins.py" or
                    relative.parts[0] not in {"jit","_dynamo","_inductor","export","compiler","package"})
            if path.is_relative_to(stdlib_root):
                relative=path.relative_to(stdlib_root)
                return relative.parts[0] not in (forbidden_stdlib-{"importlib"})|{"site-packages","dist-packages"}
            return False

        def native_origin(function):
            module=getattr(function,"__module__",None)
            if type(module) is str:
                if module.startswith(("torch._ops.","torch.ops.")):
                    return module.split(".")[2] in {"aten","prim","prims","mkldnn"}
                root=module.split(".",1)[0]
                if root in approved_names:
                    return True
            owner=getattr(function,"__self__",None)
            owner_type=type(owner)
            if owner is None:
                return False
            if owner_type is types.ModuleType:
                return owner.__name__.split(".",1)[0] in approved_names
            if owner_type in (torch.Tensor,torch.nn.Parameter):
                return getattr(function,"__name__","") not in {"numpy","__dlpack__"}
            return owner_type.__module__.split(".",1)[0] in approved_names

        def approved_operator(frame):
            if frame.f_code.co_filename != str(package/"_ops.py") or frame.f_code.co_name not in {"__call__","redispatch"}:
                return True
            try:
                owner=frame.f_locals.get("self")
                if type(owner) not in (torch._ops.OpOverload,torch._ops.OpOverloadPacket):
                    return False
                namespace=object.__getattribute__(owner,"__dict__")
                schema=namespace.get("_schema")
                name=schema.name if schema is not None else namespace.get("_qualified_op_name")
                return type(name) is str and name.split("::",1)[0] in {"aten","prim","prims","mkldnn"}
            except Exception:
                return False

        roles={proposal["sites"][role]:role for role in proposal["sites"]}
        def tensor_sequence(value):
            values=value if type(value) in (list,tuple) else [value]
            if not values or len(values)>policy["max_samples"]:
                raise ValueError("metric_tensor_sequence_capacity")
            return [snapshot(item) for item in values]
        def profile(frame,event,arg):
            nonlocal observing,counter
            if observing:
                return
            if event=="call" and not python_origin(frame.f_code.co_filename):
                unknown("unapproved_python_computation_source")
            if event=="call" and not approved_operator(frame):
                unknown("unapproved_torch_operator_namespace")
            if event=="c_call" and not native_origin(arg):
                unknown("unapproved_native_computation_source")
            site=next((i for i,s in enumerate(sites) if frame.f_code.co_filename==s["path"] and frame.f_code.co_qualname==s["qualname"] and frame.f_code.co_firstlineno==s["firstlineno"]),None)
            if site is None or site not in roles:
                return
            observing=True
            try:
                if _code_sha(frame.f_code)!=sites[site]["code_sha256"]:
                    raise ValueError("runtime_function_code_changed")
                role=roles[site]
                if event=="call":
                    if len(report["calls"])+len(stack)>=policy["max_events"]:
                        raise ValueError("inference_event_capacity")
                    names=frame.f_code.co_varnames[:frame.f_code.co_argcount+frame.f_code.co_kwonlyargcount]
                    args={name:frame.f_locals[name] for name in names}
                    counter+=1
                    row={"role":role,"site":site,"invocation":counter,"order":counter,"parent_invocation":active[stack[-1]][0]["invocation"] if stack else None}
                    active[id(frame)]=(row,args)
                    stack.append(id(frame))
                    if role=="model":
                        module=args.get("self")
                        tensors=[v for k,v in args.items() if k!="self" and type(v) is torch.Tensor]
                        if (loaded.get("module") is not module or not loaded.get("call_active") or module.training
                            or torch.is_grad_enabled() or len(tensors)!=1 or batch is None
                            or tensors[0] is not batch["features"] or snapshot(tensors[0])!=batch["feature_snapshot"]
                            or module_state(module)!=state):
                            raise ValueError("actual_loaded_module_input_unbound")
                        row["model_token"]=token(module)
                        row["torch_initial_seed"]=torch.initial_seed()
                        row["batch_features"]=batch["feature_snapshot"]
                        row["batch_labels"]=batch["label_snapshot"]
                        row["batch_ids"]=batch["ids"]
                    elif role=="prediction":
                        row["input"]=snapshot(args[proposal["prediction_argument"]])
                        row["input_token"]=row["input"]["token"]
                        model=loaded.get("final_forward")
                        if model is None or not model.get("module_call_completed") or model["output"]!=row["input"] or batch is None:
                            raise ValueError("prediction_current_model_invocation_unbound")
                        row["model_invocation"]=model["invocation"]
                        row["batch"]=dict(batch)
                    elif role=="metric":
                        row["predictions"]=tensor_sequence(args[proposal["metric_prediction_argument"]])
                        row["labels"]=tensor_sequence(args[proposal["metric_label_argument"]])
                elif event=="return" and id(frame) in active:
                    row,args=active.pop(id(frame))
                    stack.remove(id(frame))
                    if role=="getitem":
                        keys=[k for k in args if k!="self"]
                        if len(keys)!=1 or type(args[keys[0]]) is not int or type(arg) not in (tuple,list,dict):
                            raise ValueError("unsupported_getitem_contract")
                        identifier=select(arg,proposal["sample_id_selector"])
                        if type(identifier) not in (str,int) or identifier=="":
                            raise ValueError("actual_sample_id_unavailable")
                        row.update(ordinal=args[keys[0]],id=identifier,
                            feature_snapshot=snapshot(select(arg,proposal["sample_features_selector"])),
                            label=select(arg,proposal["sample_label_selector"]))
                        row["features"]=row["feature_snapshot"]["values"]
                        if type(row["label"]) not in (int,float) or not math.isfinite(row["label"]):
                            raise ValueError("getitem_label_not_exact_scalar")
                        pending.append(row)
                    elif role in ("model","prediction"):
                        out=snapshot(arg)
                        row["output_token"]=out["token"]
                        row["output"]=out
                        if role=="model" and module_state(loaded["module"])!=state:
                            raise ValueError("model_state_changed_during_inference")
                        if role=="model":
                            loaded["final_forward"]=row
                        if role=="prediction":
                            completed.append((row.pop("batch"),row,out))
                    elif role=="metric":
                        if type(arg) not in (int,float) or not math.isfinite(arg):
                            raise ValueError("metric_return_not_finite_builtin")
                        report["metric_value"]=arg
                        row["value"]=arg
                        if len(row["predictions"])!=len(completed) or len(row["labels"])!=len(completed):
                            raise ValueError("whole_metric_batch_denominator_unobserved")
                        for batch_index,(current,prediction,out) in enumerate(completed):
                            if row["predictions"][batch_index]!=out or row["labels"][batch_index]!=current["label_snapshot"]:
                                raise ValueError("whole_metric_argument_identity_unobserved")
                            model=next(r for r in report["calls"] if r["invocation"]==prediction["model_invocation"])
                            if not model.get("module_call_completed"):
                                raise ValueError("module_call_final_return_unobserved")
                            for index,item in enumerate(current["rows"]):
                                ordinal=item["ordinal"]
                                original=nn["rows"][ordinal]
                                if (item["features"]!=current["feature_snapshot"]["values"][index]
                                    or type(item["id"]) is not type(original["id"]) or item["id"]!=original["id"]
                                    or item["label"]!=original["label"] or current["label_snapshot"]["values"][index]!=original["label"]):
                                    raise ValueError("actual_sample_or_batch_changed")
                                item["label_batch_token"]=current["label_snapshot"]["token"]
                                report["samples"].append({**original,"batch_index":batch_index,"row_index":index,"getitem":item["invocation"],"model":model["invocation"],"prediction":prediction["invocation"],"metric":row["invocation"],"prediction_value":out["values"][index]})
                        completed.clear()
                    report["calls"].append(row)
            except Exception as exc:
                unknown(str(exc) if type(exc) is ValueError else "capture_failed_"+type(exc).__name__)
            finally:
                observing=False
        sys.setprofile(profile)
        author_environment()
        report["execution"]={"status":"running"}
        author_started=True
        runpy.run_path(config["entry"],run_name="__main__")
        if active or completed:
            report["unresolved"].append("incomplete_actual_call_chain")
        report["execution"]={"status":"completed"}
        report["framework"]["package_manifest_sha256_after"]=framework_sha()
    except BaseException as exc:
        if not author_started:
            report["unresolved"].append("observation_startup_unavailable_"+type(exc).__name__)
            restore_observation()
            try:
                author_environment()
                author_started=True
                report["execution"]={"status":"running"}
                runpy.run_path(config["entry"],run_name="__main__")
                report["execution"]={"status":"completed"}
            except BaseException as author_exc:
                failure=author_exc
                report["execution"]={"status":"system_exit" if isinstance(author_exc,SystemExit) else "raised","type":type(author_exc).__name__}
        else:
            failure=exc
            report["execution"]={"status":"system_exit" if isinstance(exc,SystemExit) else "raised","type":type(exc).__name__}
    finally:
        restore_observation()
        try:
            report["source_hashes_after"]={"/app/"+key:digest_file("/app/"+key) for key in config["inference"]["source_sha256"]}
            report["partition_sha256_after"]=digest_file("/app/"+config["inference"]["proposal"]["dataset_path"])
            report["checkpoint_sha256_after"]=digest_file("/app/"+config["inference"]["proposal"]["checkpoint_path"])
        except Exception:
            report["unresolved"].append("post_run_source_identity_unavailable")
        try:
            raw=json.dumps(report,allow_nan=False,sort_keys=True).encode()
            if len(raw)>config["inference"]["policy"]["max_report_bytes"]:
                report={"version":VERSION,"unresolved":["report_capacity"],"execution":report["execution"]}
                raw=json.dumps(report).encode()
            with open(config["output"],"xb") as stream:
                stream.write(raw)
        except Exception as exc:
            audit_failure=exc
    if failure is not None:
        raise failure
    if audit_failure is not None:
        raise audit_failure
    return report


if __name__=="__main__":
    try:
        with open(sys.argv[1],encoding="utf-8") as stream:
            config=json.load(stream)
        run(config)
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        raise SystemExit(1) from None
