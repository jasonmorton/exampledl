use ethereum_types::Address;
use foundry_evm::executor::{fork::MultiFork, Backend, ExecutorBuilder};
use halo2_curves::bn256::{Bn256, Fq, Fr, G1Affine};

use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{floor_planner::V1, Layouter, Value},
    dev::MockProver,
    plonk::{
        create_proof, keygen_pk, keygen_vk, verify_proof, Advice, Any, Circuit, Column,
        ConstraintSystem, Error, Fixed, Instance, ProvingKey, VerifyingKey,
    },
    poly::{
        commitment::{Params, ParamsProver},
        kzg::{
            commitment::{KZGCommitmentScheme, ParamsKZG},
            multiopen::{ProverGWC, VerifierGWC},
            strategy::AccumulatorStrategy,
        },
        Rotation, VerificationStrategy,
    },
    transcript::{TranscriptReadBuffer, TranscriptWriterBuffer},
};
use itertools::Itertools;
use plonk_verifier::{
    loader::evm::{encode_calldata, EvmLoader, EvmTranscript},
    protocol::halo2::{compile, Config},
    scheme::kzg::{AccumulationScheme, PlonkAccumulationScheme, SameCurveAccumulation},
    util::TranscriptRead,
};
use rand::{rngs::OsRng, RngCore};
use std::{iter, rc::Rc};

// use halo2_proofs::{
// //    arithmetic::FieldExt,
//   //  circuit::{Layouter, SimpleFloorPlanner},
//   //  plonk::{Circuit, Column, ConstraintSystem, Error, Instance},
// };

use halo2deeplearning::{
    affine1d::{Affine1dConfig, RawParameters},
    eltwise::{DivideBy, NonlinConfig1d, ReLu},
    inputlayer::InputConfig,
};
use std::marker::PhantomData;

// A columnar ReLu MLP consisting of a stateless MLPConfig, and an MLPCircuit with parameters and input.

#[derive(Clone)]
struct MLPConfig<F: FieldExt, const LEN: usize, const INBITS: usize, const OUTBITS: usize>
where
    [(); LEN + 3]:,
{
    input: InputConfig<F, LEN>,
    l0: Affine1dConfig<F, LEN, LEN>,
    l1: NonlinConfig1d<F, LEN, INBITS, OUTBITS, ReLu<F>>,
    l2: Affine1dConfig<F, LEN, LEN>,
    l3: NonlinConfig1d<F, LEN, INBITS, OUTBITS, ReLu<F>>,
    l4: NonlinConfig1d<F, LEN, INBITS, OUTBITS, DivideBy<F, 128>>,
    public_output: Column<Instance>,
}

#[derive(Clone)]
struct MLPCircuit<F: FieldExt, const LEN: usize, const INBITS: usize, const OUTBITS: usize> {
    // Given the stateless MLPConfig type information, a DNN trace is determined by its input and the parameters of its layers.
    // Computing the trace still requires a forward pass. The intermediate activations are stored only by the layouter.
    input: Vec<i32>,
    l0_params: RawParameters<LEN, LEN>,
    l2_params: RawParameters<LEN, LEN>,
    _marker: PhantomData<F>,
}

impl<F: FieldExt, const LEN: usize, const INBITS: usize, const OUTBITS: usize> Circuit<F>
    for MLPCircuit<F, LEN, INBITS, OUTBITS>
where
    [(); LEN + 3]:,
{
    type Config = MLPConfig<F, LEN, INBITS, OUTBITS>;
    type FloorPlanner = V1;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
        let num_advices = LEN + 3;
        println!("num_advices: {}", num_advices);
        let advices = (0..num_advices)
            .map(|_| {
                let col = cs.advice_column();
                cs.enable_equality(col);
                col
            })
            .collect::<Vec<_>>();

        let input = InputConfig::<F, LEN>::configure(cs, advices[LEN]);

        let l0 = Affine1dConfig::<F, LEN, LEN>::configure(
            cs,
            (&advices[..LEN]).try_into().unwrap(), // wts gets several col, others get a column each
            advices[LEN],                          // input
            advices[LEN + 1],                      // output
            advices[LEN + 2],                      // bias
        );

        let l1: NonlinConfig1d<F, LEN, INBITS, OUTBITS, ReLu<F>> =
            NonlinConfig1d::configure(cs, (&advices[..LEN]).try_into().unwrap());

        let l2 = Affine1dConfig::<F, LEN, LEN>::configure(
            cs,
            (&advices[..LEN]).try_into().unwrap(),
            advices[LEN],
            advices[LEN + 1],
            advices[LEN + 2],
        );

        let l3: NonlinConfig1d<F, LEN, INBITS, OUTBITS, ReLu<F>> =
            NonlinConfig1d::configure(cs, (&advices[..LEN]).try_into().unwrap());

        let l4: NonlinConfig1d<F, LEN, INBITS, OUTBITS, DivideBy<F, 128>> =
            NonlinConfig1d::configure(cs, (&advices[..LEN]).try_into().unwrap());

        let public_output: Column<Instance> = cs.instance_column();
        cs.enable_equality(public_output);

        MLPConfig {
            input,
            l0,
            l1,
            l2,
            l3,
            l4,
            public_output,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let x = config.input.layout(&mut layouter, self.input.clone())?;
        let x = config.l0.layout(
            &mut layouter,
            self.l0_params.weights.clone(),
            self.l0_params.biases.clone(),
            x,
        )?;
        let x = config.l1.layout(&mut layouter, x)?;
        let x = config.l2.layout(
            &mut layouter,
            self.l2_params.weights.clone(),
            self.l2_params.biases.clone(),
            x,
        )?;
        let x = config.l3.layout(&mut layouter, x)?;
        let x = config.l4.layout(&mut layouter, x)?;
        for i in 0..LEN {
            layouter.constrain_instance(x[i].cell(), config.public_output, i)?;
        }
        Ok(())
    }
}

// Proof construction helpers
fn sample_srs() -> ParamsKZG<Bn256> {
    ParamsKZG::<Bn256>::setup(15, OsRng)
}

fn sample_pk<C: Circuit<Fr>>(params: &ParamsKZG<Bn256>, circuit: &C) -> ProvingKey<G1Affine> {
    let vk = keygen_vk(params, circuit).unwrap();
    keygen_pk(params, vk, circuit).unwrap()
}

fn sample_proof<C: Circuit<Fr>>(
    params: &ParamsKZG<Bn256>,
    pk: &ProvingKey<G1Affine>,
    circuit: C,
    instances: Vec<Vec<Fr>>,
) -> Vec<u8> {
    MockProver::run(params.k(), &circuit, instances.clone())
        .unwrap()
        .assert_satisfied();

    let instances = instances
        .iter()
        .map(|instances| instances.as_slice())
        .collect_vec();
    let proof = {
        let mut transcript = TranscriptWriterBuffer::<_, G1Affine, _>::init(Vec::new());
        create_proof::<KZGCommitmentScheme<Bn256>, ProverGWC<_>, _, _, EvmTranscript<_, _, _, _>, _>(
            params,
            pk,
            &[circuit],
            &[instances.as_slice()],
            OsRng,
            &mut transcript,
        )
        .unwrap();
        transcript.finalize()
    };

    let accept = {
        let mut transcript = TranscriptReadBuffer::<_, G1Affine, _>::init(proof.as_slice());
        VerificationStrategy::<_, VerifierGWC<_>>::finalize(
            verify_proof::<_, VerifierGWC<_>, _, EvmTranscript<_, _, _, _>, _>(
                params.verifier_params(),
                pk.get_vk(),
                AccumulatorStrategy::new(params.verifier_params()),
                &[instances.as_slice()],
                &mut transcript,
            )
            .unwrap(),
        )
    };
    assert!(accept);

    proof
}

fn evm_verifier_codegen(
    params: &ParamsKZG<Bn256>,
    vk: &VerifyingKey<G1Affine>,
    instances: Vec<Vec<Fr>>,
) -> Vec<u8> {
    const LIMBS: usize = 4;
    const BITS: usize = 68;

    let protocol = compile(
        vk,
        Config {
            zk: true,
            query_instance: false,
            num_instance: instances
                .iter()
                .map(|instances| instances.len())
                .collect_vec(),
            num_proof: 1,
            accumulator_indices: None,
        },
    );

    let loader = EvmLoader::new::<Fq, Fr>();
    let mut transcript = EvmTranscript::<_, Rc<EvmLoader>, _, _>::new(loader.clone());
    let instances = instances
        .iter()
        .map(|instance| {
            iter::repeat_with(|| transcript.read_scalar().unwrap())
                .take(instance.len())
                .collect_vec()
        })
        .collect_vec();

    let mut strategy = SameCurveAccumulation::<_, _, LIMBS, BITS>::default();
    PlonkAccumulationScheme::accumulate(
        &protocol,
        &loader,
        instances,
        &mut transcript,
        &mut strategy,
    )
    .unwrap();
    strategy.finalize(params.get_g()[0], params.g2(), params.s_g2());
    loader.deployment_code()
}

fn evm_verify(deployment_code: Vec<u8>, instances: Vec<Vec<Fr>>, proof: Vec<u8>) {
    let calldata = encode_calldata(instances, proof);
    let success = {
        let mut evm = ExecutorBuilder::default()
            .with_gas_limit(u64::MAX.into())
            .build(Backend::new(MultiFork::new().0, None));

        let caller = Address::from_low_u64_be(0xfe);
        let verifier = evm
            .deploy(caller, deployment_code.into(), 0.into(), None)
            .unwrap()
            .address;
        let result = evm
            .call_raw(caller, verifier, calldata.into(), 0.into())
            .unwrap();

        !result.reverted
    };
    assert!(success);
}

pub fn mlprun() {
    use halo2_curves::bn256::Fr as F;
    //    use halo2_curves::pasta::Fp as F;
    use halo2_proofs::dev::MockProver;
    use halo2deeplearning::fieldutils::i32tofelt;

    //    let k = 15; //2^k rows  15
    // parameters
    let l0weights: Vec<Vec<i32>> = vec![
        vec![10, 0, 0, -1],
        vec![0, 10, 1, 0],
        vec![0, 1, 10, 0],
        vec![1, 0, 0, 10],
    ];
    let l0biases: Vec<i32> = vec![0, 0, 0, 1];
    let l0_params = RawParameters {
        weights: l0weights,
        biases: l0biases,
    };
    let l2weights: Vec<Vec<i32>> = vec![
        vec![0, 3, 10, -1],
        vec![0, 10, 1, 0],
        vec![0, 1, 0, 12],
        vec![1, -2, 32, 0],
    ];
    let l2biases: Vec<i32> = vec![12, 14, 17, 1];
    let l2_params = RawParameters {
        weights: l2weights,
        biases: l2biases,
    };
    // input data
    let input: Vec<i32> = vec![-30, -21, 11, 40];

    let circuit = MLPCircuit::<F, 4, 14, 14> {
        //14,14
        input,
        l0_params,
        l2_params,
        _marker: PhantomData,
    };

    let public_input: Vec<i32> = unsafe {
        vec![
            (531f32 / 128f32).round().to_int_unchecked::<i32>(),
            (103f32 / 128f32).round().to_int_unchecked::<i32>(),
            (4469f32 / 128f32).round().to_int_unchecked::<i32>(),
            (2849f32 / 128f32).to_int_unchecked::<i32>(),
        ]
    };

    // Mock Proof
    // let prover = MockProver::run(
    //     k,
    //     &circuit,
    //     vec![public_input.iter().map(|x| i32tofelt::<F>(*x)).collect()],
    //     //            vec![vec![(4).into(), (1).into(), (35).into(), (22).into()]],
    // )
    // .unwrap();
    // prover.assert_satisfied();

    println!("public input {:?}", public_input);

    let params = sample_srs();

    //    let circuit = StandardPlonk::rand(OsRng);
    let pk = sample_pk(&params, &circuit);
    //  let deployment_code = evm_verifier_codegen(&params, pk.get_vk(), circuit.instances());
    let instances = vec![public_input.iter().map(|x| i32tofelt::<F>(*x)).collect()];
    let deployment_code = evm_verifier_codegen(&params, pk.get_vk(), instances.clone());

    println!("Deployment code is {:?} bytes", deployment_code.len());

    let proof = sample_proof(&params, &pk, circuit.clone(), instances.clone());
    println!("Verifying");
    evm_verify(deployment_code, instances, proof);
}
