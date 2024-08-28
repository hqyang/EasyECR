from typing import Dict
from typing import Union
from typing import Any
from typing import Optional
from collections import defaultdict

from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
import spacy
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel
from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import LongformerModel
from transformers import LongformerPreTrainedModel
from transformers import AdamW
from transformers import get_scheduler
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor
from pytorch_lightning import seed_everything

from easyecr.ecr_data.data_structure.data_structure import EcrData
from easyecr.ecr_data.data_structure.data_structure import Mention
from easyecr.ecr_data.data_structure.data_structure import Event
from easyecr.ecr_model.model.pl_ecr_models.pl_ecr_model import PlEcrModel


class GlobalLocalTopicCoreferenceDataset(Dataset):
    def __init__(self, ecr_data: EcrData):
        self.ecr_data = ecr_data
        self.index_type = ecr_data.meta["index_type"]

        self.documents = self.ecr_data.documents
        self.events = self.ecr_data.events
        self.mentions = self.ecr_data.mentions

        if "kbp" in ecr_data.name:
            self.kbp_event_type_init()
            self.data = self.adapt_kbp_data()
        elif "ace" in ecr_data.name:
            self.ace_event_type_init()
            self.data = self.adapt_ace_data()
        elif "maven" in ecr_data.name:
            self.maven_event_type_init()
            self.data = self.adapt_maven_data()
        else:
            raise NotImplementedError

    def kbp_event_type_init(self):
        self.subtypes = [  # 18 subtypes
            "artifact",
            "transferownership",
            "transaction",
            "broadcast",
            "contact",
            "demonstrate",
            "injure",
            "transfermoney",
            "transportartifact",
            "attack",
            "meet",
            "elect",
            "endposition",
            "correspondence",
            "arrestjail",
            "startposition",
            "transportperson",
            "die",
        ]
        self.vocab = [
            "have",
            "said",
            "has",
            "'s",
            "had",
            "did",
            "him",
            "think",
            "get",
            "know",
            "does",
            "'m",
            "see",
            "being",
            "going",
            "say",
            "go",
            "make",
            "made",
            "got",
            "want",
            "am",
            "take",
            "'re",
            "us",
            "apple",
            "told",
            "china",
            "obama",
            "u.s.",
            "'ve",
            "need",
            "according",
            "including",
            "let",
            "called",
            "used",
            "give",
            "pay",
            "saying",
            "went",
            "keep",
            "come",
            "believe",
            "killed",
            "convicted",
            "bush",
            "found",
            "left",
            "read",
            "find",
            "done",
            "came",
            "seems",
            "put",
            "having",
            "took",
            "doing",
            "work",
            "says",
            "use",
            "given",
            "united states",
            "feel",
            "thought",
            "arrested",
            "trying",
            "syria",
            "getting",
            "uk",
            "look",
            "making",
            "happened",
            "buy",
            "help",
            "elected",
            "israel",
            "posted",
            "tell",
            "hope",
            "agree",
            "wanted",
            "like",
            "become",
            "makes",
            "stop",
            "charged",
            "seen",
            "start",
            "known",
            "sent",
            "asked",
            "reported",
            "using",
            "russia",
            "allowed",
            "set",
            "understand",
            "involved",
            "mean",
            "based",
            "started",
            "taken",
            "died",
            "try",
            "run",
            "paid",
            "mandela",
            "live",
            "iraq",
            "held",
            "iran",
            "america",
            "heard",
            "saw",
            "looking",
            "happen",
            "working",
            "support",
            "taking",
            "leave",
            "expected",
            "released",
            "seem",
            "tried",
            "xinhua",
            "wants",
            "washington",
            "sentenced",
            "appear",
            "nokia",
            "coming",
            "remember",
            "announced",
            "guess",
            "eu",
            "love",
            "worked",
            "gave",
            "accused",
            "lost",
            "pakistan",
            "gets",
            "goes",
            "continue",
            "microsoft",
            "stay",
            "began",
            "served",
            "fired",
            "met",
            "became",
            "ask",
            "pardoned",
            "talking",
            "living",
            "gone",
            "egypt",
            "india",
            "hit",
            "move",
            "brought",
            "filed",
            "care",
            "following",
            "kill",
            "hear",
            "turned",
            "knew",
            "decided",
            "agreed",
            "needs",
            "vote",
            "snowden",
            "comes",
            "call",
            "allow",
            "led",
            "new york",
            "new york times",
            "sounds",
            "received",
            "knows",
            "leaving",
            "bring",
            "running",
            "killing",
            "wait",
            "north korea",
            "send",
            "talk",
            "die",
            "meet",
            "needed",
            "thinking",
            "cyprus",
            "means",
            "win",
            "congress",
            "show",
            "caused",
            "born",
            "explain",
            "clinton",
            "calling",
            "added",
            "wish",
            "giving",
            "florida",
            "london",
            "bought",
            "ordered",
            "europe",
            "issued",
            "stand",
            "spend",
            "thank",
            "shot",
            "google",
            "hold",
            "senate",
            "forced",
            "provide",
            "white house",
            "seeing",
            "wonder",
            "change",
            "related",
            "married",
            "claimed",
            "end",
            "texas",
            "takes",
            "philippines",
            "spent",
            "deal",
            "moved",
            "wrote",
            "foxconn",
            "france",
            "looks",
            "cut",
            "telling",
            "un",
            "follow",
            "morsi",
            "sell",
            "turn",
            "speak",
            "paying",
            "passed",
            "condensed",
            "won",
            "scotland",
            "usa",
            "nelson mandela",
            "showed",
            "committed",
            "face",
            "appointed",
            "injured",
            "ruled",
            "barack obama",
            "britain",
            "ukraine",
            "lose",
            "buying",
            "hate",
            "fighting",
            "granted",
            "denied",
            "confirmed",
            "germany",
            "happens",
            "kept",
            "considered",
            "sandusky",
            "chun",
            "failed",
            "remain",
            "helped",
            "works",
            "serve",
            "gives",
            "'d",
            "affected",
            "caught",
            "afghanistan",
            "changed",
            "leading",
            "receive",
            "owned",
            "cause",
            "created",
            "supreme court",
            "starting",
            "lead",
            "believed",
            "thinks",
            "sold",
            "japan",
            "trump",
            "followed",
            "mention",
            "supposed",
            "expect",
            "serving",
            "realize",
            "planned",
            "detained",
            "refused",
            "return",
            "remains",
            "felt",
            "extradited",
            "protect",
            "doubt",
            "blame",
            "moving",
            "built",
            "appeared",
            "offered",
            "reached",
            "include",
            "speaking",
            "waiting",
            "carry",
            "declined",
            "included",
            "bangladesh",
            "istanbul",
            "create",
            "spain",
            "california",
            "beijing",
            "decide",
            "supporting",
            "helping",
            "considering",
            "provided",
            "lived",
            "avoid",
            "rejected",
            "attacked",
            "arrived",
            "executed",
            "described",
            "named",
            "voted",
            "stopped",
            "claiming",
            "vietnam",
            "shows",
            "appears",
            "ended",
            "check",
            "returned",
            "discuss",
            "asking",
            "australia",
            "learn",
            "bet",
            "knowing",
            "imagine",
            "adding",
            "prove",
            "deserve",
            "remained",
            "consider",
            "putting",
            "save",
            "watch",
            "continued",
            "fight",
            "broke",
            "accept",
            "expressed",
            "ran",
            "join",
            "selling",
            "supported",
            "wanting",
            "written",
            "watching",
            "suggest",
            "pardon",
            "holding",
            "address",
            "cairo",
            "claim",
            "learned",
            "turkey",
            "pass",
            "played",
            "united nations",
            "carried",
            "signed",
            "seeking",
            "building",
            "grow",
            "libya",
            "south africa",
            "prevent",
            "losing",
            "assad",
            "treated",
            "meant",
            "scheduled",
            "driving",
            "happening",
            "add",
            "lying",
            "sound",
            "disagree",
            "raped",
            "travel",
            "mexico",
            "joined",
            "becoming",
            "stated",
            "required",
            "defend",
            "canada",
            "seemed",
            "haiyan",
            "afford",
            "attempted",
            "samsung",
            "gotten",
            "hired",
            "steve",
            "italy",
            "nominated",
            "register",
            "cost",
            "planning",
            "shown",
            "chang",
            "sought",
            "putin",
            "missing",
            "represent",
            "cover",
            "army",
            "published",
            "declared",
            "intended",
            "worry",
            "admitted",
            "looked",
            "growing",
            "sending",
            "urged",
            "ignore",
            "steve jobs",
            "moscow",
            "quoted",
            "raised",
            "post",
            "build",
            "zimmerman",
            "muslim brotherhood",
            "brazil",
            "reporting",
            "fined",
            "retired",
            "demanding",
            "gm",
            "covered",
            "clicking",
            "reading",
            "begin",
            "forget",
            "acting",
            "hoping",
            "resigned",
            "sitting",
            "argue",
            "compared",
            "sued",
            "fuck",
            "breaking",
            "drive",
            "cia",
            "dropped",
            "walk",
            "admit",
            "middle east",
            "ensure",
            "walked",
            "england",
            "watched",
            "beat",
            "force",
            "opposed",
            "south korea",
            "lives",
            "pick",
            "regarding",
            "justice department",
        ]

        self.id2subtype = {idx: c for idx, c in enumerate(self.subtypes, start=1)}
        self.subtype2id = {v: k for k, v in self.id2subtype.items()}

    def ace_event_type_init(self):
        self.subtypes = [  # 18 subtypes
            "artifact",
            "transferownership",
            "transaction",
            "broadcast",
            "contact",
            "demonstrate",
            "injure",
            "transfermoney",
            "transportartifact",
            "attack",
            "meet",
            "elect",
            "endposition",
            "correspondence",
            "arrestjail",
            "startposition",
            "transportperson",
            "die",
        ]
        self.vocab = self.verb_entity_recog()
        self.id2subtype = {idx: c for idx, c in enumerate(self.subtypes, start=1)}
        self.subtype2id = {v: k for k, v in self.id2subtype.items()}

    def maven_event_type_init(self):
        subtype2id = {
            "Control": 41,
            "Achieve": 127,
            "Creating": 60,
            "Self_motion": 46,
            "Motion": 10,
            "Process_start": 31,
            "Process_end": 61,
            "Cause_to_amalgamate": 131,
            "Aiming": 98,
            "Bringing": 78,
            "Participation": 54,
            "Ratification": 144,
            "Conquering": 21,
            "Using": 28,
            "Deciding": 55,
            "Cause_to_make_progress": 137,
            "Dispersal": 119,
            "Coming_to_be": 29,
            "Causation": 5,
            "Damaging": 11,
            "Placing": 4,
            "Manufacturing": 81,
            "Name_conferral": 69,
            "Becoming": 126,
            "Destroying": 12,
            "Cause_change_of_strength": 129,
            "Arriving": 6,
            "Motion_directional": 77,
            "Preserving": 100,
            "Reporting": 33,
            "Killing": 20,
            "Escaping": 97,
            "Recovering": 27,
            "Warning": 2,
            "Removing": 51,
            "Reveal_secret": 104,
            "Patrolling": 113,
            "Carry_goods": 92,
            "Attack": 23,
            "Know": 1,
            "Criminal_investigation": 108,
            "Committing_crime": 72,
            "Judgment_communication": 87,
            "Coming_to_believe": 38,
            "Legal_rulings": 66,
            "Death": 13,
            "Rescuing": 86,
            "Bodily_harm": 34,
            "Statement": 36,
            "Check": 18,
            "Telling": 64,
            "Preventing_or_letting": 9,
            "Communication": 79,
            "Arrest": 114,
            "Breathing": 112,
            "Assistance": 73,
            "Giving": 68,
            "Change": 65,
            "Use_firearm": 71,
            "Body_movement": 62,
            "Wearing": 136,
            "Violence": 133,
            "Come_together": 135,
            "Publishing": 141,
            "Supporting": 42,
            "Getting": 58,
            "Request": 40,
            "Perception_active": 14,
            "Rite": 140,
            "Earnings_and_losses": 24,
            "Recording": 91,
            "Military_operation": 45,
            "Cause_change_of_position_on_a_scale": 37,
            "Risk": 160,
            "Hindering": 99,
            "Hostile_encounter": 19,
            "Reforming_a_system": 134,
            "Change_of_leadership": 96,
            "Labeling": 152,
            "Social_event": 82,
            "Change_sentiment": 122,
            "Extradition": 121,
            "Commitment": 123,
            "Defending": 43,
            "Agree_or_refuse_to_act": 53,
            "Commerce_sell": 116,
            "Expressing_publicly": 39,
            "Temporary_stay": 118,
            "Cure": 117,
            "Commerce_pay": 124,
            "Collaboration": 120,
            "Response": 105,
            "Convincing": 115,
            "Writing": 85,
            "Presence": 15,
            "Catastrophe": 3,
            "Connect": 103,
            "Departing": 94,
            "Influence": 16,
            "Supply": 130,
            "Sending": 7,
            "Hold": 89,
            "Justifying": 156,
            "Rewards_and_punishments": 148,
            "Quarreling": 75,
            "Releasing": 22,
            "Arranging": 70,
            "Action": 50,
            "Employment": 139,
            "Change_event_time": 32,
            "Surrounding": 74,
            "Being_in_operation": 90,
            "Scrutiny": 106,
            "Adducing": 142,
            "GiveUp": 95,
            "Expansion": 63,
            "Confronting_problem": 110,
            "Receiving": 17,
            "Building": 44,
            "Robbery": 83,
            "Competition": 84,
            "Legality": 138,
            "Commerce_buy": 146,
            "Scouring": 132,
            "Choosing": 25,
            "Institutionalization": 149,
            "Cause_to_be_included": 30,
            "Besieging": 59,
            "Prison": 155,
            "Bearing_arms": 67,
            "Vocalizations": 159,
            "Practice": 128,
            "Create_artwork": 101,
            "Becoming_a_member": 49,
            "Protest": 8,
            "Theft": 166,
            "Education_teaching": 56,
            "Forming_relationships": 48,
            "Traveling": 26,
            "Award": 168,
            "Revenge": 162,
            "Emptying": 57,
            "Sign_agreement": 145,
            "Suspicion": 35,
            "Submitting_documents": 154,
            "Hiding_objects": 109,
            "Testing": 150,
            "Resolve_problem": 161,
            "Terrorism": 158,
            "GetReady": 47,
            "Incident": 167,
            "Surrendering": 52,
            "Having_or_lacking_access": 165,
            "Imposing_obligation": 147,
            "Ingestion": 151,
            "Kidnapping": 153,
            "Openness": 102,
            "Exchange": 143,
            "Emergency": 157,
            "Cost": 93,
            "Research": 164,
            "Change_tool": 88,
            "Containing": 80,
            "Filling": 125,
            "Renting": 111,
            "Expend_resource": 76,
            "Lighting": 107,
            "Limiting": 163,
        }
        self.subtype2id = subtype2id
        self.id2subtype = {v: k for k, v in subtype2id.items()}
        self.vocab = self.verb_entity_recog()

    def adapt_maven_data(self):
        data = []
        doc2mentions = defaultdict(list)
        for _, mention in self.mentions.items():
            doc_id = mention.doc_id
            doc2mentions[doc_id].append(mention)
        for doc_id, doc in self.documents.items():
            doc_mentions = doc2mentions[doc_id]
            sentences = doc.meta["sentences"]
            events, event_mentions, mention_char_pos, word_dists = [], [], [], []
            for m in doc_mentions:
                events.append(
                    {
                        "event_id": m.mention_id,
                        "char_start": m.anchor.start,
                        "char_end": m.anchor.end,
                        "trigger": m.anchor.text,
                        "subtype": self.subtype2id.get(m.meta["event_type"], 0),
                        "cluster_id": m.meta["event_id"],
                        "mention_id": m.mention_id,
                    }
                )
                mention_text = m.extent.text
                event_mentions.append(mention_text)
                mention_char_pos.append([m.anchor.start - m.extent.start, m.anchor.end - m.extent.start])

                # before = sentences[m.meta["sent_id"] - 1] if m.meta["sent_id"] > 0 else ""
                # after = sentences[m.meta["sent_id"] + 1] if m.meta["sent_id"] < len(sentences) - 1 else ""
                # event_mention = before + (" " if len(before) > 0 else "") + mention_text + " " + after
                word_dists.append(self.__get_word_dist(mention_text))
            data.append(
                {
                    "id": doc_id,
                    "document": doc.text,
                    "events": events,
                    "event_mentions": event_mentions,
                    "mention_char_pos": mention_char_pos,
                    "word_dists": word_dists,
                    "index_type": self.index_type,
                }
            )
        return data

    def verb_entity_recog(self):
        nlp = spacy.load("en_core_web_sm")

        word_count = {}
        for doc in self.documents.values():
            text = doc.text
            res = nlp(text)
            for token in res:
                if token.pos_ == "VERB":
                    word_count[token.text] = word_count.get(token.text, 0) + 1
            for token in res.ents:
                word_count[token.text] = word_count.get(token.text, 0) + 1
        sorted_items = sorted(word_count.items(), key=lambda x: x[0], reverse=True)
        top_500_items = sorted_items[:500]
        vocab = list(dict(top_500_items).keys())
        return vocab

    def __get_word_dist(self, mention_text: str):
        mention_text = mention_text.lower()
        return [1 if w in mention_text else 0 for w in self.vocab]

    def adapt_kbp_data(self):
        data = []
        doc2mentions = defaultdict(list)
        for _, mention in self.mentions.items():
            doc_id = mention.doc_id
            doc2mentions[doc_id].append(mention)

        for doc_id, doc in self.documents.items():
            doc_mentions = doc2mentions[doc_id]
            sentences = doc.meta["sentences"]
            events, event_mentions, mention_char_pos, word_dists = [], [], [], []
            for m in doc_mentions:
                events.append(
                    {
                        "event_id": m.mention_id,
                        "char_start": m.anchor.start,
                        "char_end": m.anchor.end,
                        "trigger": m.anchor.text,
                        "subtype": self.subtype2id.get(m.meta["subtype"], 0),
                        "cluster_id": m.meta["event_id"],
                        "mention_id": m.mention_id,
                    }
                )

            for m in doc_mentions:
                # before = sentences[m.meta["sent_idx"] - 1]["text"] if m.meta["sent_idx"] > 0 else ""
                # after = sentences[m.meta["sent_idx"] + 1]["text"] if m.meta["sent_idx"] < len(sentences) - 1 else ""
                # event_mention = (
                #     before + (" " if len(before) > 0 else "") + sentences[m.meta["sent_idx"]]["text"] + " " + after
                # )
                event_mentions.append(m.extent.text)
                mention_char_pos.append([m.anchor.start - m.extent.start, m.anchor.end - m.extent.start])
                word_dists.append(self.__get_word_dist(m.extent.text))

            data.append(
                {
                    "id": doc_id,
                    "document": doc.text,
                    "events": events,
                    "event_mentions": event_mentions,
                    "mention_char_pos": mention_char_pos,
                    "word_dists": word_dists,
                    "index_type": self.index_type,
                }
            )
        return data

    def adapt_ace_data(self):
        data = []
        doc2mentions = defaultdict(list)
        for _, mention in self.mentions.items():
            doc_id = mention.doc_id
            doc2mentions[doc_id].append(mention)

        for doc_id, doc in self.documents.items():
            doc_mentions = doc2mentions[doc_id]
            if len(doc_mentions) == 0:
                continue

            doc_text = doc.text
            events, event_mentions, mention_char_pos, word_dists = [], [], [], []
            for m in doc_mentions:
                events.append(
                    {
                        "event_id": m.mention_id,
                        "char_start": m.anchor.start,
                        "char_end": m.anchor.end,
                        "trigger": m.anchor.text,
                        "subtype": self.subtype2id.get(m.meta["subtype"], 0),
                        "cluster_id": m.meta["event_id"],
                        "mention_id": m.mention_id,
                    }
                )
                mention_text = m.extent.text
                event_mentions.append(mention_text)
                mention_char_pos.append([m.anchor.start - m.extent.start, m.anchor.end - m.extent.start])
                # before, after = self.get_local_mention(doc_text, m.extent.start, m.extent.end, self.index_type)
                # event_mention = before + (" " if len(before) > 0 else "") + mention_text + " " + after
                word_dists.append(self.__get_word_dist(mention_text))
            data.append(
                {
                    "id": doc_id,
                    "document": doc.text,
                    "events": events,
                    "event_mentions": event_mentions,
                    "mention_char_pos": mention_char_pos,
                    "word_dists": word_dists,
                    "index_type": self.index_type,
                }
            )
        return data

    def get_local_mention(self, doc_text: Union[list, str], mention_start: int, mention_end: int, index_type: str):
        if index_type == "word":
            len_extent = 50
            before, after = [], []
            if mention_start > len_extent:
                for item in doc_text[mention_start - len_extent : mention_start]:
                    before.append(item["text"])
            else:
                for item in doc_text[:mention_start]:
                    before.append(item["text"])

            for item in doc_text[mention_end : mention_end + len_extent]:
                after.append(item["text"])
            return " ".join(before), " ".join(after)
        elif index_type == "char":
            len_extent = 300
            before, after = "", ""
            before = (
                doc_text[mention_start - len_extent : mention_start]
                if mention_start > len_extent
                else doc_text[:mention_start]
            )
            after = doc_text[mention_end : mention_end + len_extent]
            return before, after
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class LongformerSoftmaxForECwithMaskTopic(LongformerPreTrainedModel):
    def __init__(self, conf):
        config = AutoConfig.from_pretrained(conf["doc_encoder"])
        mention_config = AutoConfig.from_pretrained(conf["mention_encoder"])
        super().__init__(config)
        self.validation_step_outputs = []
        self.num_labels = 2
        self.dist_dim = 500
        self.num_subtypes = conf["num_classes"]
        self.topic_dim = 32
        self.hidden_size = config.hidden_size + mention_config.hidden_size + self.topic_dim
        self.mention_encoder_dim = mention_config.hidden_size

        # SimpleTopicVMFModel
        self.batch_size = 1
        self.original_dim = 500
        self.latent_dim = 32
        self.inter_dim = 64
        self.fc_h = nn.Linear(self.original_dim, self.inter_dim)
        self.encoder_act = nn.ReLU()
        self.fc_mu = nn.Linear(self.inter_dim, self.latent_dim)
        self.decoder_h = nn.Linear(self.latent_dim, self.inter_dim)
        self.decoder_act_1 = nn.ReLU()
        self.decoder_mean = nn.Linear(self.inter_dim, self.original_dim)
        self.decoder_act_2 = nn.Sigmoid()
        self.W = self._init_w(self.latent_dim)

        # encoder & pooler
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=config.hidden_size)
        self.mention_encoder = BertModel.from_pretrained(conf["mention_encoder"])
        self.mention_dropout = nn.Dropout(mention_config.hidden_dropout_prob)
        self.mention_span_extractor = SelfAttentiveSpanExtractor(input_dim=self.mention_encoder_dim)
        self.subtype_classifier = nn.Linear(self.mention_encoder_dim, self.num_subtypes)

        self.cosine_space_dim, self.cosine_slices, self.tensor_factor = (64, 128, 4)
        self.cosine_mat_p = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_slices), requires_grad=True))
        self.cosine_mat_q = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_space_dim), requires_grad=True))
        self.cosine_ffnn = nn.Linear(self.hidden_size, self.cosine_space_dim)
        self.coref_classifier = nn.Linear(3 * self.hidden_size + self.cosine_slices, self.num_labels)
        self.post_init()

    # SimpleTopicVMFModel
    def _init_w(self, dims, kappa=20):
        epsilon = 1e-7
        x = np.arange(-1 + epsilon, 1, epsilon)
        y = kappa * x + np.log(1 - x**2) * (dims - 3) / 2
        y = np.cumsum(np.exp(y - y.max()))
        y = y / y[-1]
        W = torch.tensor((np.interp(np.random.random(10**6), y, x)), dtype=torch.float32)
        return W

    # SimpleTopicVMFModel
    def _encode(self, inputs):
        result = self.fc_h(inputs)
        result = self.encoder_act(result)
        mu = self.fc_mu(result)
        return F.normalize(mu, p=2, dim=-1)

    # SimpleTopicVMFModel
    def _decode(self, z):
        result = self.decoder_h(z)
        result = self.decoder_act_1(result)
        result = self.decoder_mean(result)
        result = self.decoder_act_2(result)
        return result

    # SimpleTopicVMFModel
    def _sampling(self, mu):
        self.W = self.W.to(self.use_device)
        batch_size, dims = mu.size()
        # real-time sampling w
        idx = torch.randint(0, 10**6, size=(batch_size,), device=self.use_device)
        w = torch.gather(self.W, 0, idx)
        w = torch.repeat_interleave(w.unsqueeze(-1), repeats=dims, dim=-1)
        # real-time sampling z
        eps = torch.randn_like(mu)
        nu = eps - torch.sum(eps * mu, dim=1, keepdim=True) * mu
        nu = F.normalize(nu, p=2, dim=-1)
        return w * mu + (1 - w**2) ** 0.5 * nu

    def topic_model(self, batch_e1_inputs, batch_e2_inputs, batch_mask=None):
        z_e1_mean = self._encode(batch_e1_inputs)
        z_e1_mean = z_e1_mean.view(-1, self.latent_dim)
        z_e1 = self._sampling(z_e1_mean)
        z_e1 = z_e1.view(self.batch_size, -1, self.latent_dim)
        batch_e1_recons = self._decode(z_e1)
        z_e2_mean = self._encode(batch_e2_inputs)
        z_e2_mean = z_e2_mean.view(-1, self.latent_dim)
        z_e2 = self._sampling(z_e2_mean)
        z_e2 = z_e2.view(self.batch_size, -1, self.latent_dim)
        batch_e2_recons = self._decode(z_e2)
        # calculate loss
        if batch_mask is not None:
            active_loss = batch_mask.view(-1) == 1
            active_e1_inputs = batch_e1_inputs.view(-1, self.original_dim)[active_loss]
            active_e1_recons = batch_e1_recons.view(-1, self.original_dim)[active_loss]
            active_e2_inputs = batch_e2_inputs.view(-1, self.original_dim)[active_loss]
            active_e2_recons = batch_e2_recons.view(-1, self.original_dim)[active_loss]
        else:
            active_e1_inputs = batch_e1_inputs.view(-1, self.original_dim)
            active_e1_recons = batch_e1_recons.view(-1, self.original_dim)
            active_e2_inputs = batch_e2_inputs.view(-1, self.original_dim)
            active_e2_recons = batch_e2_recons.view(-1, self.original_dim)
        e1_recons_loss = F.mse_loss(active_e1_recons, active_e1_inputs)
        e2_recons_loss = F.mse_loss(active_e2_recons, active_e2_inputs)
        loss = 0.5 * e1_recons_loss + 0.5 * e2_recons_loss
        return loss, z_e1, z_e2

    def _multi_cosine(self, batch_event_1_reps, batch_event_2_reps):
        batch_event_1_reps = self.cosine_ffnn(batch_event_1_reps)
        batch_event_1_reps = batch_event_1_reps.unsqueeze(dim=2)
        batch_event_1_reps = self.cosine_mat_q * batch_event_1_reps
        batch_event_1_reps = batch_event_1_reps.permute((0, 1, 3, 2))
        batch_event_1_reps = torch.matmul(batch_event_1_reps, self.cosine_mat_p)
        batch_event_1_reps = batch_event_1_reps.permute((0, 1, 3, 2))
        # vector normalization
        norms_1 = (batch_event_1_reps**2).sum(axis=-1, keepdims=True) ** 0.5
        batch_event_1_reps = batch_event_1_reps / norms_1

        batch_event_2_reps = self.cosine_ffnn(batch_event_2_reps)
        batch_event_2_reps = batch_event_2_reps.unsqueeze(dim=2)
        batch_event_2_reps = self.cosine_mat_q * batch_event_2_reps
        batch_event_2_reps = batch_event_2_reps.permute((0, 1, 3, 2))
        batch_event_2_reps = torch.matmul(batch_event_2_reps, self.cosine_mat_p)
        batch_event_2_reps = batch_event_2_reps.permute((0, 1, 3, 2))
        # vector normalization
        norms_2 = (batch_event_2_reps**2).sum(axis=-1, keepdims=True) ** 0.5
        batch_event_2_reps = batch_event_2_reps / norms_2
        return torch.sum(batch_event_1_reps * batch_event_2_reps, dim=-1)

    def _cal_circle_loss(self, event_1_reps, event_2_reps, coref_labels, l=20.0):
        norms_1 = (event_1_reps**2).sum(axis=1, keepdims=True) ** 0.5
        event_1_reps = event_1_reps / norms_1
        norms_2 = (event_2_reps**2).sum(axis=1, keepdims=True) ** 0.5
        event_2_reps = event_2_reps / norms_2
        event_cos = torch.sum(event_1_reps * event_2_reps, dim=1) * l
        # calculate the difference between each pair of Cosine values
        event_cos_diff = event_cos[:, None] - event_cos[None, :]
        # find (noncoref, coref) index
        select_idx = coref_labels[:, None] < coref_labels[None, :]
        select_idx = select_idx.float()
        event_cos_diff = event_cos_diff - (1 - select_idx) * 1e12
        event_cos_diff = event_cos_diff.view(-1)
        event_cos_diff = torch.cat((torch.tensor([0.0], device=self.use_device), event_cos_diff), dim=0)
        return torch.logsumexp(event_cos_diff, dim=0)

    def forward(
        self,
        batch_inputs,
        batch_events,
        batch_mention_inputs_with_mask,
        batch_mention_events,
        batch_event_dists,
        batch_event_cluster_ids=None,
        batch_event_subtypes=None,
    ):
        outputs = self.longformer(**batch_inputs)
        self.use_device = self.longformer.device

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # construct local event mask representations
        batch_local_event_mask_reps = []
        for mention_mask_inputs, mention_events in zip(batch_mention_inputs_with_mask, batch_mention_events):
            encoder_outputs = self.mention_encoder(**mention_mask_inputs)
            mention_mask_output = encoder_outputs[0]
            mention_mask_output = self.mention_dropout(mention_mask_output)
            mention_event_list = [[event] for event in mention_events]
            mention_event_list = torch.tensor(mention_event_list, device=self.use_device)
            local_event_mask_reps = self.mention_span_extractor(mention_mask_output, mention_event_list).squeeze(
                dim=1
            )  # (event_num, dim)
            batch_local_event_mask_reps.append(local_event_mask_reps)
        # construct event pairs (event_1, event_2)
        batch_event_1_list, batch_event_2_list = [], []
        batch_local_event_1_mask_reps, batch_local_event_2_mask_reps = [], []
        batch_event_1_dists, batch_event_2_dists = [], []
        max_len, batch_event_mask = 0, []
        if batch_event_cluster_ids is not None:
            batch_coref_labels = []
            batch_local_event_1_subtypes, batch_local_event_2_subtypes = [], []
            for events, local_event_mask_reps, event_dists, event_cluster_ids, event_subtypes in zip(
                batch_events,
                batch_local_event_mask_reps,
                batch_event_dists,
                batch_event_cluster_ids,
                batch_event_subtypes,
            ):
                event_1_list, event_2_list = [], []
                event_1_idx, event_2_idx = [], []
                coref_labels = []
                event_1_subtypes, event_2_subtypes = [], []
                for i in range(len(events) - 1):
                    for j in range(i + 1, len(events)):
                        event_1_list.append(events[i])
                        event_2_list.append(events[j])
                        event_1_idx.append(i)
                        event_2_idx.append(j)
                        cluster_id_1, cluster_id_2 = event_cluster_ids[i], event_cluster_ids[j]
                        coref_labels.append(1 if cluster_id_1 == cluster_id_2 else 0)
                        event_1_subtypes.append(event_subtypes[i])
                        event_2_subtypes.append(event_subtypes[j])
                max_len = max(max_len, len(coref_labels))
                batch_event_1_list.append(event_1_list)
                batch_event_2_list.append(event_2_list)
                batch_local_event_1_mask_reps.append(
                    torch.index_select(local_event_mask_reps, 0, torch.tensor(event_1_idx).to(self.use_device))
                )
                batch_local_event_2_mask_reps.append(
                    torch.index_select(local_event_mask_reps, 0, torch.tensor(event_2_idx).to(self.use_device))
                )
                batch_event_1_dists.append(
                    torch.index_select(event_dists, 0, torch.tensor(event_1_idx).to(self.use_device))
                )
                batch_event_2_dists.append(
                    torch.index_select(event_dists, 0, torch.tensor(event_2_idx).to(self.use_device))
                )
                batch_local_event_1_subtypes.append(event_1_subtypes)
                batch_local_event_2_subtypes.append(event_2_subtypes)
                batch_coref_labels.append(coref_labels)
                batch_event_mask.append([1] * len(coref_labels))
            # padding
            for b_idx in range(len(batch_coref_labels)):
                pad_length = max_len - len(batch_coref_labels[b_idx]) if max_len > 0 else 1
                batch_event_1_list[b_idx] += [[0, 0]] * pad_length
                batch_event_2_list[b_idx] += [[0, 0]] * pad_length
                batch_local_event_1_mask_reps[b_idx] = torch.cat(
                    [
                        batch_local_event_1_mask_reps[b_idx],
                        torch.zeros((pad_length, self.mention_encoder_dim)).to(self.use_device),
                    ],
                    dim=0,
                ).unsqueeze(0)
                batch_local_event_2_mask_reps[b_idx] = torch.cat(
                    [
                        batch_local_event_2_mask_reps[b_idx],
                        torch.zeros((pad_length, self.mention_encoder_dim)).to(self.use_device),
                    ],
                    dim=0,
                ).unsqueeze(0)
                batch_event_1_dists[b_idx] = torch.cat(
                    [batch_event_1_dists[b_idx], torch.zeros((pad_length, self.dist_dim)).to(self.use_device)], dim=0
                ).unsqueeze(0)
                batch_event_2_dists[b_idx] = torch.cat(
                    [batch_event_2_dists[b_idx], torch.zeros((pad_length, self.dist_dim)).to(self.use_device)], dim=0
                ).unsqueeze(0)
                batch_local_event_1_subtypes[b_idx] += [0] * pad_length
                batch_local_event_2_subtypes[b_idx] += [0] * pad_length
                batch_coref_labels[b_idx] += [0] * pad_length
                batch_event_mask[b_idx] += [0] * pad_length
        else:
            for events, local_event_mask_reps, event_dists in zip(
                batch_events, batch_local_event_mask_reps, batch_event_dists
            ):
                event_1_list, event_2_list = [], []
                event_1_idx, event_2_idx = [], []
                for i in range(len(events) - 1):
                    for j in range(i + 1, len(events)):
                        event_1_list.append(events[i])
                        event_2_list.append(events[j])
                        event_1_idx.append(i)
                        event_2_idx.append(j)
                max_len = max(max_len, len(event_1_list))
                batch_event_1_list.append(event_1_list)
                batch_event_2_list.append(event_2_list)
                batch_local_event_1_mask_reps.append(
                    torch.index_select(local_event_mask_reps, 0, torch.tensor(event_1_idx).to(self.use_device))
                )
                batch_local_event_2_mask_reps.append(
                    torch.index_select(local_event_mask_reps, 0, torch.tensor(event_2_idx).to(self.use_device))
                )
                batch_event_1_dists.append(
                    torch.index_select(event_dists, 0, torch.tensor(event_1_idx).to(self.use_device))
                )
                batch_event_2_dists.append(
                    torch.index_select(event_dists, 0, torch.tensor(event_2_idx).to(self.use_device))
                )
                batch_event_mask.append([1] * len(event_1_list))
            # padding
            for b_idx in range(len(batch_event_mask)):
                pad_length = max_len - len(batch_event_mask[b_idx]) if max_len > 0 else 1
                batch_event_1_list[b_idx] += [[0, 0]] * pad_length
                batch_event_2_list[b_idx] += [[0, 0]] * pad_length
                batch_local_event_1_mask_reps[b_idx] = torch.cat(
                    [
                        batch_local_event_1_mask_reps[b_idx],
                        torch.zeros((pad_length, self.mention_encoder_dim)).to(self.use_device),
                    ],
                    dim=0,
                ).unsqueeze(0)
                batch_local_event_2_mask_reps[b_idx] = torch.cat(
                    [
                        batch_local_event_2_mask_reps[b_idx],
                        torch.zeros((pad_length, self.mention_encoder_dim)).to(self.use_device),
                    ],
                    dim=0,
                ).unsqueeze(0)
                batch_event_1_dists[b_idx] = torch.cat(
                    [batch_event_1_dists[b_idx], torch.zeros((pad_length, self.dist_dim)).to(self.use_device)], dim=0
                ).unsqueeze(0)
                batch_event_2_dists[b_idx] = torch.cat(
                    [batch_event_2_dists[b_idx], torch.zeros((pad_length, self.dist_dim)).to(self.use_device)], dim=0
                ).unsqueeze(0)
                batch_event_mask[b_idx] += [0] * pad_length
        # extract events
        batch_event_1 = torch.tensor(batch_event_1_list).to(self.use_device)
        batch_event_2 = torch.tensor(batch_event_2_list).to(self.use_device)
        batch_mask = torch.tensor(batch_event_mask).to(self.use_device)
        batch_event_1_reps = self.span_extractor(sequence_output, batch_event_1, span_indices_mask=batch_mask)
        batch_event_2_reps = self.span_extractor(sequence_output, batch_event_2, span_indices_mask=batch_mask)
        # predict event subtype
        batch_local_event_1_mask_reps = torch.cat(batch_local_event_1_mask_reps, dim=0)
        batch_local_event_2_mask_reps = torch.cat(batch_local_event_2_mask_reps, dim=0)
        event_1_subtypes_logits = self.subtype_classifier(batch_local_event_1_mask_reps)
        event_2_subtypes_logits = self.subtype_classifier(batch_local_event_2_mask_reps)
        # generate event topics
        batch_event_1_dists = torch.cat(batch_event_1_dists, dim=0)
        batch_event_2_dists = torch.cat(batch_event_2_dists, dim=0)

        loss_topic, batch_e1_topics, batch_e2_topics = self.topic_model(
            batch_event_1_dists, batch_event_2_dists, batch_mask
        )
        # matching & predict coref
        batch_event_1_reps = torch.cat([batch_event_1_reps, batch_local_event_1_mask_reps, batch_e1_topics], dim=-1)
        batch_event_2_reps = torch.cat([batch_event_2_reps, batch_local_event_2_mask_reps, batch_e2_topics], dim=-1)
        batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
        batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
        batch_seq_reps = torch.cat(
            [batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi, batch_multi_cosine], dim=-1
        )

        logits = self.coref_classifier(batch_seq_reps)
        # calculate loss
        loss, batch_labels = None, None
        if batch_event_cluster_ids is not None and max_len > 0:
            loss_subtype_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            active_loss = batch_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            batch_labels = torch.tensor(batch_coref_labels).to(self.use_device)
            active_labels = batch_labels.view(-1)[active_loss]
            active_e_1_sutype_logits = event_1_subtypes_logits.view(-1, self.num_subtypes)[active_loss]
            active_e_2_sutype_logits = event_2_subtypes_logits.view(-1, self.num_subtypes)[active_loss]
            active_subtype_logits = torch.cat([active_e_1_sutype_logits, active_e_2_sutype_logits], dim=0)
            batch_event_1_subtypes = torch.tensor(batch_local_event_1_subtypes).to(self.use_device)
            batch_event_2_subtypes = torch.tensor(batch_local_event_2_subtypes).to(self.use_device)
            active_e_1_subtypes = batch_event_1_subtypes.view(-1)[active_loss]
            active_e_2_subtypes = batch_event_2_subtypes.view(-1)[active_loss]
            active_subtype_labels = torch.cat([active_e_1_subtypes, active_e_2_subtypes], dim=0)

            loss_subtype = loss_subtype_fct(active_subtype_logits, active_subtype_labels)
            loss_coref = loss_subtype_fct(active_logits, active_labels)
            active_event_1_reps = batch_event_1_reps.view(-1, self.hidden_size)[active_loss]
            active_event_2_reps = batch_event_2_reps.view(-1, self.hidden_size)[active_loss]
            loss_contrasive = self._cal_circle_loss(active_event_1_reps, active_event_2_reps, active_labels)
            loss = (
                torch.log(1 + loss_coref)
                + torch.log(1 + loss_subtype)
                + torch.log(1 + loss_topic)
                + 0.2 * loss_contrasive
            )
            # loss = torch.log(1 + loss_coref) + torch.log(1 + loss_subtype) + torch.log(1 + loss_topic)
        return loss, logits, batch_mask, batch_labels


class GlobalLocalTopicCoreferenceModule(pl.LightningModule):
    def __init__(self, conf, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(conf)
        self.validation_step_outputs = []

        self.ecr_model = LongformerSoftmaxForECwithMaskTopic(conf)

    def forward(
        self,
        batch_inputs,
        batch_events,
        batch_mention_inputs_with_mask,
        batch_mention_events,
        batch_event_dists,
        batch_event_cluster_ids=None,
        batch_event_subtypes=None,
        batch_mention_ids=None,
    ) -> Any:
        result = {}
        if len(batch_mention_inputs_with_mask) == 0:
            loss = None
            result["loss"] = loss
        else:
            loss, logits, batch_mask, batch_labels = self.ecr_model.forward(
                batch_inputs,
                batch_events,
                batch_mention_inputs_with_mask,
                batch_mention_events,
                batch_event_dists,
                batch_event_cluster_ids,
                batch_event_subtypes,
            )
            logits = torch.softmax(logits, dim=-1)
            similarities = logits[:, :, 1]
            distances = 1 - similarities
            result["distances"] = distances
            result["loss"] = loss
        torch.cuda.empty_cache()

        # distances_square = torch.square(distances)

        # result["labels"] = batch_labels

        # result["distances_square"] = distances_square
        # result["logits"] = logits
        return result

    def training_step(self, batch, batch_idx):
        if isinstance(batch, list):
            batch = batch[0]
        forward_output = self.forward(**batch)
        if forward_output["loss"] is not None:
            return forward_output["loss"]
        else:
            return None

    def on_train_epoch_end(self):
        torch.cuda.empty_cache()

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, list):
            batch = batch[0]
        forward_output = self.forward(**batch)
        if forward_output["loss"] is not None:
            return forward_output["loss"]
        else:
            return None

    def on_validation_epoch_end(self):
        torch.cuda.empty_cache()

    def get_optimizer_and_scheduler(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {"params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        if self.hparams.optimizer == "adam":
            optimizer = AdamW(
                optimizer_grouped_parameters,
                self.hparams.learning_rate,
                betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
                eps=self.hparams.adam_epsilon,
            )
        else:
            raise NotImplementedError

        lr_scheduler = get_scheduler("linear", optimizer, num_warmup_steps=250, num_training_steps=300000)
        return optimizer, lr_scheduler

    def configure_optimizers(self):
        optimizer, lr_scheduler = self.get_optimizer_and_scheduler()
        return [optimizer], [lr_scheduler]

    def predict_step(self, batch, batch_idx):
        if isinstance(batch, list):
            batch = batch[0]
        result = self(**batch)
        return result


class GlobalLocalTopicModel(PlEcrModel):
    def __init__(
        self,
        mode: str,
        model_dir: str,
        model_filename: str,
        trainer_parameters: Dict[str, Any],
        best_model_filename: str = "best.ckpt",
        conf: Optional[DictConfig] = None,
    ):
        super().__init__(mode, model_dir, model_filename, trainer_parameters, best_model_filename, conf)
        seed_everything(42)

    def build_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.conf["module"]["doc_encoder"])
        self.mention_tokenizer = AutoTokenizer.from_pretrained(self.conf["module"]["mention_encoder"])

    def instanciate_module(self):
        module = GlobalLocalTopicCoreferenceModule(self.conf["module"])
        return module

    def load_module(self, filepath: str):
        result = GlobalLocalTopicCoreferenceModule.load_from_checkpoint(filepath)
        return result

    def cut_sent(self, sent, e_char_start, e_char_end, max_length):
        max_length = 4096
        span_max_length = (max_length - 50) // 2
        before = " ".join([c for c in sent[:e_char_start].split(" ") if c != ""][-span_max_length:]).strip()
        trigger = sent[e_char_start : e_char_end + 1]
        after = " ".join([c for c in sent[e_char_end + 1 :].split(" ") if c != ""][:span_max_length]).strip()
        return (
            before + " " + trigger + " " + after,
            len(before) + 1,
            len(before) + len(trigger),
        )

    def collate_fn(self, batch_samples):
        tokenizer = self.tokenizer
        mention_tokenizer = self.mention_tokenizer

        batch_sentences, batch_events = [], []
        batch_mentions, batch_mention_pos = [], []
        batch_event_dists = []
        index_type = ""
        doc_id = None
        for sample in batch_samples:
            doc_id = sample["id"]
            batch_sentences.append(sample["document"])
            batch_events.append(sample["events"])
            batch_mentions.append(sample["event_mentions"])
            batch_mention_pos.append(sample["mention_char_pos"])
            batch_event_dists.append(sample["word_dists"])
            index_type = sample["index_type"]
        batch_inputs = tokenizer(
            batch_sentences,
            max_length=4096,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        batch_filtered_events = []
        batch_filtered_mention_inputs_with_mask = []
        batch_filtered_mention_events = []
        batch_filtered_event_dists = []
        batch_filtered_event_cluster_id = []
        batch_filtered_event_subtypes = []
        batch_filtered_mention_ids = []
        for sentence, events, mentions, mention_poss, event_dists in zip(
            batch_sentences,
            batch_events,
            batch_mentions,
            batch_mention_pos,
            batch_event_dists,
        ):
            encoding = tokenizer(sentence, max_length=4096, truncation=True)
            filtered_events = []
            filtered_event_mentions = []
            filtered_mention_events = []
            filtered_event_dists = []
            filtered_event_cluster_id = []
            filtered_event_subtypes = []
            filtered_mention_ids = []
            for event, mention, mention_pos, dist in zip(events, mentions, mention_poss, event_dists):
                if index_type == "char":
                    token_start = encoding.char_to_token(event["char_start"])
                    if not token_start:
                        token_start = encoding.char_to_token(event["char_start"] + 1)
                    token_end = encoding.char_to_token(event["char_end"])
                    if not token_start or not token_end:
                        continue
                elif index_type == "word":
                    token_start = encoding.word_to_tokens(event["char_start"])
                    if not token_start:
                        token_start = encoding.word_to_tokens(event["char_start"] + 1)
                    token_end = encoding.word_to_tokens(event["char_end"])
                    if not token_start or not token_end:
                        continue
                    token_start = token_start[0]
                    token_end = token_end[1]
                else:
                    raise NotImplementedError

                mention_char_start, mention_char_end = mention_pos
                # mention, mention_char_start, mention_char_end = self.cut_sent(
                #     mention,
                #     mention_char_start,
                #     mention_char_end,
                #     4096,
                # )
                # assert mention[mention_char_start : mention_char_end + 1] == event["trigger"]
                mention_encoding = mention_tokenizer(mention, max_length=4096, truncation=True)
                if index_type == "char":
                    mention_token_start = mention_encoding.char_to_token(mention_char_start)
                    if not mention_token_start:
                        mention_token_start = mention_encoding.char_to_token(mention_char_start + 1)
                    mention_token_end = mention_encoding.char_to_token(mention_char_end)
                    # assert mention_token_start and mention_token_end
                    if not mention_token_start or not mention_token_end:
                        continue
                elif index_type == "word":
                    mention_token_start = mention_encoding.word_to_tokens(mention_char_start)
                    if not mention_token_start:
                        mention_token_start = mention_encoding.word_to_tokens(mention_char_start + 1)
                    mention_token_end = mention_encoding.word_to_tokens(mention_char_end)
                    assert mention_token_start and mention_token_end
                    mention_token_start = mention_token_start[0]
                    mention_token_end = mention_token_end[1]
                else:
                    raise NotImplementedError
                filtered_events.append([token_start, token_end])
                filtered_event_mentions.append(mention)
                filtered_mention_events.append([mention_token_start, mention_token_end])
                filtered_event_dists.append(dist)
                filtered_event_cluster_id.append(event["cluster_id"])
                filtered_event_subtypes.append(event["subtype"])
                filtered_mention_ids.append(event["mention_id"])
            batch_filtered_events.append(filtered_events)
            if len(filtered_event_mentions) == 0:
                pass
            else:
                batch_filtered_mention_inputs_with_mask.append(
                    mention_tokenizer(
                        filtered_event_mentions,
                        max_length=4096,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    )
                )
            batch_filtered_mention_events.append(filtered_mention_events)
            batch_filtered_event_dists.append(np.asarray(filtered_event_dists))
            batch_filtered_event_cluster_id.append(filtered_event_cluster_id)
            batch_filtered_event_subtypes.append(filtered_event_subtypes)
            batch_filtered_mention_ids.append(filtered_mention_ids)
        for b_idx in range(len(batch_filtered_event_subtypes)):
            for e_idx, (e_start, e_end) in enumerate(batch_filtered_mention_events[b_idx]):
                batch_filtered_mention_inputs_with_mask[b_idx]["input_ids"][e_idx][
                    e_start : e_end + 1
                ] = mention_tokenizer.mask_token_id

        batch_data = {
            "batch_inputs": {k_: v_ for k_, v_ in batch_inputs.items()},
            "batch_events": batch_filtered_events,
            "batch_mention_inputs_with_mask": [
                {k_: v_ for k_, v_ in inputs.items()} for inputs in batch_filtered_mention_inputs_with_mask
            ],
            "batch_mention_events": batch_filtered_mention_events,
            "batch_event_dists": [
                torch.tensor(event_dists, dtype=torch.float32) for event_dists in batch_filtered_event_dists
            ],
            "batch_event_cluster_ids": batch_filtered_event_cluster_id,
            "batch_event_subtypes": batch_filtered_event_subtypes,
            "batch_mention_ids": batch_filtered_mention_ids,
        }
        return batch_data

    def prepare_data(self, data: EcrData, mode: str):
        if data is None:
            return None, None
        dataset = GlobalLocalTopicCoreferenceDataset(data)

        batch_size = self.conf["dataloader"][f"{mode}_batch_size"]
        num_workers = self.conf["dataloader"]["num_workers"]
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            shuffle=(mode == "train"),
            pin_memory=True,
        )
        dataloaders = [dataloader]
        return dataset, dataloaders

    def inner_pred(self, trainer, module, dataloaders, dataset):
        predictions = trainer.predict(module, dataloaders=dataloaders)
        distances = []
        mix_doc_index = []
        for idx, prediction in enumerate(predictions):
            if prediction.get("distances") == None:
                mix_doc_index.append(idx)
            else:
                distances.extend(prediction["distances"].numpy().tolist())

        datas = dataset.data
        result = []
        for doc_i, data in enumerate(datas):
            if doc_i in mix_doc_index:
                continue
            doc_data = self.collate_fn([data])
            id_id_distance = defaultdict(dict)
            events = doc_data["batch_mention_ids"]

            event_1_list, event_2_list = [], []
            if len(events[0]) == 1:
                mention_id1 = events[0][0]
                mention_id2 = events[0][0]
                id_id_distance[mention_id1][mention_id2] = 1.0
                id_id_distance[mention_id2][mention_id1] = 1.0
            else:
                for i in range(len(events[0]) - 1):
                    for j in range(i + 1, len(events[0])):
                        event_1_list.append(events[0][i])
                        event_2_list.append(events[0][j])
                for pair_j, pair in enumerate(zip(event_1_list, event_2_list)):
                    distance = distances[doc_i][pair_j]
                    mention_id1 = pair[0]
                    mention_id2 = pair[1]
                    if mention_id1 not in id_id_distance:
                        id_id_distance[mention_id1] = {}
                    if mention_id2 not in id_id_distance:
                        id_id_distance[mention_id2] = {}
                    id_id_distance[mention_id1][mention_id2] = distance
                    id_id_distance[mention_id2][mention_id1] = distance
            result.append(id_id_distance)
        return result

    def predict(self, data: EcrData, output_tag: str) -> EcrData:
        trainer = pl.Trainer(
            accelerator="gpu",
            # devices=args.devices
        )

        test_dataset, test_dataloaders = self.prepare_data(data, mode="predict")
        result = self.inner_pred(trainer, self.module, test_dataloaders, test_dataset)
        documents = {}
        doc_mentions = {}
        events = []
        for id_id_distance in result:
            for mention_id, id_distance in id_id_distance.items():
                mention = data.mentions[mention_id]
                mention.add_tag(output_tag, id_distance)
                doc_mentions[mention_id] = mention
                doc_id = mention.doc_id
                document = data.documents[doc_id]
                documents[doc_id] = document

        event2mentions = defaultdict(list)
        for _, mention in doc_mentions.items():
            event_id = mention.meta["event_id"]
            event2mentions[event_id].append(mention)
        for event_id, mentions in event2mentions.items():
            event = Event(event_id, mentions)
            events.append(event)

        new_data = EcrData(data.name, documents, doc_mentions, events, data.meta)
        return new_data

    def get_predict_type(self) -> str:
        result = Mention.mention_distance_tag_name
        return result
